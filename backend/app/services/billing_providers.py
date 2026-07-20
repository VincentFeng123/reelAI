"""Stripe Checkout, Portal, and webhook adapter for ReelAI billing."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from ..db import execute_modify, fetch_one
from .billing import (
    PRODUCTION_ENVIRONMENT,
    SANDBOX_ENVIRONMENT,
    TERMINAL_SUBSCRIPTION_STATUSES,
    billing_entitlement_environment,
    billing_status,
    plan_for_product,
    provider_customer_id,
    record_provider_event,
    stripe_price_for_plan,
    subscription_rows,
    upsert_provider_customer,
    upsert_subscription,
)


class BillingConfigurationError(RuntimeError):
    pass


class DuplicateSubscriptionError(RuntimeError):
    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"An existing {provider} subscription must be managed there.")


class BillingVerificationError(ValueError):
    pass


class BillingAccountNotFoundError(RuntimeError):
    pass


def _obj_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _id(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(_obj_get(value, "id", "") or "")


def _epoch_seconds_iso(value: Any) -> str | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def _stripe_event_environment(event: Any) -> str:
    livemode = _obj_get(event, "livemode")
    if livemode is True:
        return PRODUCTION_ENVIRONMENT
    if livemode is False:
        return SANDBOX_ENVIRONMENT
    raise BillingVerificationError("Stripe event is missing a valid livemode value.")


def _stripe_module():
    try:
        import stripe
    except ImportError as exc:  # pragma: no cover - dependency is pinned in production
        raise BillingConfigurationError("The Stripe SDK is not installed.") from exc
    secret = os.getenv("STRIPE_SECRET_KEY", "").strip()
    if not secret:
        raise BillingConfigurationError("STRIPE_SECRET_KEY is not configured.")
    stripe.api_key = secret
    return stripe


def _stripe_configured_environment() -> str:
    secret = os.getenv("STRIPE_SECRET_KEY", "").strip().lower()
    if secret.startswith(("sk_live_", "rk_live_")):
        environment = PRODUCTION_ENVIRONMENT
    elif secret.startswith(("sk_test_", "rk_test_")):
        environment = SANDBOX_ENVIRONMENT
    else:
        raise BillingConfigurationError(
            "STRIPE_SECRET_KEY must be a Stripe test or live secret key."
        )
    if environment != billing_entitlement_environment():
        raise BillingConfigurationError(
            "Stripe key mode must match BILLING_ENTITLEMENT_ENVIRONMENT."
        )
    return environment


def _web_origin() -> str:
    configured = (
        os.getenv("BILLING_WEB_ORIGIN", "").strip()
        or os.getenv("FRONTEND_ORIGIN", "").strip()
    )
    if not configured:
        raise BillingConfigurationError("BILLING_WEB_ORIGIN is not configured.")
    parsed = urlsplit(configured)
    try:
        parsed.port
    except ValueError as exc:
        raise BillingConfigurationError(
            "BILLING_WEB_ORIGIN contains an invalid port."
        ) from exc
    hostname = str(parsed.hostname or "").lower()
    is_local_http = parsed.scheme == "http" and hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }
    if (
        (parsed.scheme != "https" and not is_local_http)
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise BillingConfigurationError(
            "BILLING_WEB_ORIGIN must be an absolute HTTPS origin "
            "(HTTP is allowed only for localhost)."
        )
    return configured.rstrip("/")


def _nonterminal_stripe_subscription(conn: Any, account_id: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in subscription_rows(conn, account_id)
            if str(row.get("provider") or "") == "stripe"
            and str(row.get("status") or "").strip().lower()
            not in TERMINAL_SUBSCRIPTION_STATUSES
        ),
        None,
    )


def lock_billing_account(conn: Any, account_id: str) -> None:
    """Serialize Checkout and deletion for one existing ReelAI account."""
    clean_account_id = str(account_id or "").strip()
    if not clean_account_id:
        raise BillingAccountNotFoundError("The ReelAI account no longer exists.")
    if isinstance(conn, sqlite3.Connection):
        found = execute_modify(
            conn,
            "UPDATE community_accounts SET updated_at = updated_at WHERE id = ?",
            (clean_account_id,),
        )
    else:
        found = bool(
            fetch_one(
                conn,
                "SELECT id FROM community_accounts WHERE id = ? FOR UPDATE",
                (clean_account_id,),
            )
        )
    if not found:
        raise BillingAccountNotFoundError("The ReelAI account no longer exists.")


def _remote_nonterminal_stripe_subscription(stripe: Any, customer_id: str) -> Any | None:
    subscriptions = stripe.Subscription.list(
        customer=customer_id,
        status="all",
        limit=100,
    )
    return next(
        (
            subscription
            for subscription in _obj_get(subscriptions, "data", []) or []
            if str(_obj_get(subscription, "status", "unknown") or "unknown")
            .strip()
            .lower()
            not in TERMINAL_SUBSCRIPTION_STATUSES
        ),
        None,
    )


def _open_stripe_checkout_url(
    stripe: Any,
    customer_id: str,
    *,
    plan_code: str,
) -> str | None:
    sessions = stripe.checkout.Session.list(
        customer=customer_id,
        status="open",
        limit=100,
    )
    matching_url: str | None = None
    for session in _obj_get(sessions, "data", []) or []:
        if str(_obj_get(session, "mode", "") or "") != "subscription":
            continue
        metadata = _obj_get(session, "metadata", {}) or {}
        session_plan = str(_obj_get(metadata, "plan", "") or "").strip().lower()
        url = str(_obj_get(session, "url", "") or "").strip()
        if session_plan == plan_code and url and matching_url is None:
            matching_url = url
            continue
        session_id = _id(session)
        if session_id:
            stripe.checkout.Session.expire(session_id)
    return matching_url


def create_stripe_checkout(
    conn: Any,
    *,
    account: dict[str, Any],
    plan_code: str,
) -> str:
    account_id = str(account.get("id") or "")
    lock_billing_account(conn, account_id)
    if _nonterminal_stripe_subscription(conn, account_id):
        raise DuplicateSubscriptionError("Stripe")

    stripe = _stripe_module()
    provider_environment = _stripe_configured_environment()
    customer_id = provider_customer_id(
        conn,
        account_id,
        "stripe",
        provider_environment=provider_environment,
    )
    if not customer_id:
        customer = stripe.Customer.create(
            email=str(account.get("email") or "").strip() or None,
            metadata={"account_id": account_id},
            idempotency_key=f"reelai-customer-{account_id}",
        )
        customer_id = _id(customer)
        if not customer_id:
            raise RuntimeError("Stripe did not return a customer id.")
        upsert_provider_customer(
            conn,
            account_id=account_id,
            provider="stripe",
            provider_environment=provider_environment,
            external_customer_id=customer_id,
        )

    try:
        price_id = stripe_price_for_plan(plan_code)
    except RuntimeError as exc:
        raise BillingConfigurationError(str(exc)) from exc
    origin = _web_origin()
    if _remote_nonterminal_stripe_subscription(stripe, customer_id):
        raise DuplicateSubscriptionError("Stripe")
    open_checkout_url = _open_stripe_checkout_url(
        stripe,
        customer_id,
        plan_code=plan_code,
    )
    if open_checkout_url:
        return open_checkout_url
    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=f"{origin}/?settings=plan&checkout=success",
        cancel_url=f"{origin}/?settings=plan&checkout=cancelled",
        client_reference_id=account_id,
        metadata={"account_id": account_id, "plan": plan_code},
        subscription_data={"metadata": {"account_id": account_id}},
        allow_promotion_codes=False,
        idempotency_key=(
            f"reelai-checkout-{provider_environment.lower()}-"
            f"{account_id}-{uuid.uuid4().hex}"
        ),
    )
    url = str(_obj_get(session, "url", "") or "")
    if not url:
        raise RuntimeError("Stripe did not return a Checkout URL.")
    return url


def create_stripe_portal(conn: Any, *, account_id: str) -> str:
    provider_environment = _stripe_configured_environment()
    customer_id = provider_customer_id(
        conn,
        account_id,
        "stripe",
        provider_environment=provider_environment,
    )
    if not customer_id:
        raise ValueError("No Stripe billing account exists for this ReelAI account.")
    stripe = _stripe_module()
    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=f"{_web_origin()}/?settings=plan",
    )
    url = str(_obj_get(session, "url", "") or "")
    if not url:
        raise RuntimeError("Stripe did not return a Customer Portal URL.")
    return url


def _stripe_subscription_product(subscription: Any) -> str:
    items = _obj_get(_obj_get(subscription, "items", {}), "data", []) or []
    first = items[0] if items else {}
    price = _obj_get(first, "price", {})
    return _id(price)


def _stripe_subscription_period_end(subscription: Any) -> str | None:
    """Support both legacy and current Stripe subscription period shapes."""
    legacy_period_end = _epoch_seconds_iso(
        _obj_get(subscription, "current_period_end")
    )
    if legacy_period_end:
        return legacy_period_end
    items = _obj_get(_obj_get(subscription, "items", {}), "data", []) or []
    item_period_ends = [
        period_end
        for item in items
        if (period_end := _epoch_seconds_iso(_obj_get(item, "current_period_end")))
    ]
    return max(item_period_ends, default=None)


def _stripe_subscription_account_id(
    conn: Any,
    subscription: Any,
    *,
    provider_environment: str,
    fallback_account_id: str = "",
) -> str:
    metadata = _obj_get(subscription, "metadata", {}) or {}
    account_id = str(_obj_get(metadata, "account_id", "") or fallback_account_id).strip()
    if account_id:
        return account_id
    customer_id = _id(_obj_get(subscription, "customer", ""))
    row = fetch_one(
        conn,
        "SELECT account_id FROM billing_provider_customers "
        "WHERE provider = 'stripe' AND provider_environment = ? "
        "AND external_customer_id = ?",
        (provider_environment, customer_id),
    )
    return str((row or {}).get("account_id") or "").strip()


def _stripe_invoice_for_charge(stripe: Any, charge: Any) -> Any | None:
    """Resolve an invoice across legacy and current Stripe payment shapes."""
    invoice_id = _id(_obj_get(charge, "invoice", ""))
    if not invoice_id:
        payment_intent_id = _id(_obj_get(charge, "payment_intent", ""))
        if payment_intent_id:
            invoice_payments = stripe.InvoicePayment.list(
                payment={
                    "type": "payment_intent",
                    "payment_intent": payment_intent_id,
                },
                limit=1,
            )
            payments = _obj_get(invoice_payments, "data", []) or []
            if payments:
                invoice_id = _id(_obj_get(payments[0], "invoice", ""))
    return stripe.Invoice.retrieve(invoice_id) if invoice_id else None


def _stripe_invoice_subscription_id(invoice: Any) -> str:
    legacy_subscription_id = _id(_obj_get(invoice, "subscription", ""))
    if legacy_subscription_id:
        return legacy_subscription_id
    parent = _obj_get(invoice, "parent", {}) or {}
    subscription_details = _obj_get(parent, "subscription_details", {}) or {}
    return _id(_obj_get(subscription_details, "subscription", ""))


def _store_stripe_subscription(
    conn: Any,
    subscription: Any,
    *,
    event_created_at: str,
    provider_environment: str,
    fallback_account_id: str = "",
    override_status: str | None = None,
) -> dict[str, Any]:
    external_id = _id(subscription)
    customer_id = _id(_obj_get(subscription, "customer", ""))
    account_id = _stripe_subscription_account_id(
        conn,
        subscription,
        provider_environment=provider_environment,
        fallback_account_id=fallback_account_id,
    )
    if not external_id or not account_id:
        raise BillingVerificationError("Stripe subscription is missing ReelAI account binding.")
    if not fetch_one(conn, "SELECT id FROM community_accounts WHERE id = ?", (account_id,)):
        raise BillingVerificationError("Stripe subscription references an unknown account.")
    existing_binding = fetch_one(
        conn,
        "SELECT account_id, provider_environment FROM billing_subscriptions "
        "WHERE provider = 'stripe' AND external_subscription_id = ?",
        (external_id,),
    )
    if existing_binding:
        if str(existing_binding.get("account_id") or "") != account_id:
            raise BillingVerificationError(
                "Provider subscription is already bound to another ReelAI account."
            )
        existing_environment = str(
            existing_binding.get("provider_environment") or ""
        )
        if existing_environment != provider_environment:
            raise BillingVerificationError(
                "Provider subscription cannot change billing environment."
            )
    if customer_id:
        try:
            upsert_provider_customer(
                conn,
                account_id=account_id,
                provider="stripe",
                provider_environment=provider_environment,
                external_customer_id=customer_id,
            )
        except ValueError as exc:
            raise BillingVerificationError(str(exc)) from exc
    existing = fetch_one(
        conn,
        "SELECT plan_code, external_product_id FROM billing_subscriptions "
        "WHERE provider = 'stripe' AND provider_environment = ? "
        "AND external_subscription_id = ?",
        (provider_environment, external_id),
    )
    product_id = _stripe_subscription_product(subscription)
    try:
        plan_code = plan_for_product("stripe", product_id)
    except RuntimeError as exc:
        stored_product_id = str((existing or {}).get("external_product_id") or "")
        if not existing or (product_id and product_id != stored_product_id):
            raise BillingConfigurationError(str(exc)) from exc
        plan_code = str(existing.get("plan_code") or "")
        product_id = product_id or stored_product_id
    if not plan_code and existing:
        stored_product_id = str(existing.get("external_product_id") or "")
        if product_id and product_id != stored_product_id:
            raise BillingVerificationError(
                "Stripe subscription uses an unconfigured Price ID."
            )
        plan_code = str(existing.get("plan_code") or "")
        product_id = product_id or stored_product_id
    if not plan_code:
        raise BillingVerificationError("Stripe subscription uses an unconfigured Price ID.")
    status = override_status or str(_obj_get(subscription, "status", "unknown") or "unknown")
    try:
        return upsert_subscription(
            conn,
            account_id=account_id,
            provider="stripe",
            external_subscription_id=external_id,
            external_product_id=product_id,
            plan_code=plan_code,
            status=status,
            current_period_end=_stripe_subscription_period_end(subscription),
            provider_environment=provider_environment,
            cancel_at_period_end=bool(
                _obj_get(subscription, "cancel_at_period_end", False)
            ),
            provider_event_created_at=event_created_at,
        )
    except ValueError as exc:
        raise BillingVerificationError(str(exc)) from exc


def construct_stripe_event(payload: bytes, signature: str) -> Any:
    stripe = _stripe_module()
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
    if not webhook_secret:
        raise BillingConfigurationError("STRIPE_WEBHOOK_SECRET is not configured.")
    try:
        return stripe.Webhook.construct_event(payload, signature, webhook_secret)
    except Exception as exc:
        signature_error = getattr(
            getattr(stripe, "error", None),
            "SignatureVerificationError",
            None,
        )
        if isinstance(signature_error, type) and isinstance(exc, signature_error):
            raise BillingVerificationError("Stripe webhook signature is invalid.") from exc
        raise


def process_stripe_event(conn: Any, event: Any) -> bool:
    """Process and deduplicate one verified Stripe event inside a DB transaction."""
    stripe = _stripe_module()
    event_id = _id(event)
    event_type = str(_obj_get(event, "type", "") or "")
    event_created_at = _epoch_seconds_iso(_obj_get(event, "created")) or datetime.now(
        timezone.utc
    ).isoformat()
    if not event_id or not event_type:
        raise BillingVerificationError("Stripe event is missing id or type.")
    if not isinstance(conn, sqlite3.Connection):
        digest = hashlib.sha256(f"stripe:{event_id}".encode("utf-8")).digest()
        event_lock_id = int.from_bytes(digest[:8], "big", signed=True)
        fetch_one(
            conn,
            "SELECT pg_advisory_xact_lock(?) AS acquired",
            (event_lock_id,),
        )
    if fetch_one(
        conn,
        "SELECT external_event_id FROM billing_provider_events WHERE provider = 'stripe' AND external_event_id = ?",
        (event_id,),
    ):
        return False

    obj = _obj_get(_obj_get(event, "data", {}), "object", {})
    if event_type == "checkout.session.completed":
        mode = str(_obj_get(obj, "mode", "") or "")
        subscription_id = _id(_obj_get(obj, "subscription", ""))
        if mode == "subscription" and subscription_id:
            account_id = str(
                _obj_get(obj, "client_reference_id", "")
                or _obj_get(_obj_get(obj, "metadata", {}) or {}, "account_id", "")
                or ""
            ).strip()
            subscription = stripe.Subscription.retrieve(subscription_id)
            try:
                _store_stripe_subscription(
                    conn,
                    subscription,
                    event_created_at=event_created_at,
                    provider_environment=_stripe_event_environment(event),
                    fallback_account_id=account_id,
                )
            except BillingVerificationError as exc:
                if "unknown account" not in str(exc):
                    raise
                # A Checkout completion can race account deletion after the
                # deletion path expires the open Session. Never leave that
                # verified, orphaned subscription billing remotely.
                _cancel_remote_stripe_subscription(stripe, subscription_id)
    elif event_type in {
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    }:
        override = "canceled" if event_type.endswith(".deleted") else None
        try:
            _store_stripe_subscription(
                conn,
                obj,
                event_created_at=event_created_at,
                provider_environment=_stripe_event_environment(event),
                override_status=override,
            )
        except BillingVerificationError:
            # Account deletion cancels Stripe before removing the local account.
            # Its asynchronous deletion event can arrive after the cascade; it
            # must be acknowledged without recreating the deleted entitlement.
            if not event_type.endswith(".deleted"):
                raise
    elif event_type == "charge.refunded":
        amount = int(_obj_get(obj, "amount", 0) or 0)
        amount_refunded = int(_obj_get(obj, "amount_refunded", 0) or 0)
        if amount > 0 and amount_refunded >= amount:
            invoice = _stripe_invoice_for_charge(stripe, obj)
            subscription_id = _stripe_invoice_subscription_id(invoice)
            if subscription_id:
                subscription = stripe.Subscription.retrieve(subscription_id)
                subscription_status = str(
                    _obj_get(subscription, "status", "unknown") or "unknown"
                ).strip().lower()
                if subscription_status not in TERMINAL_SUBSCRIPTION_STATUSES:
                    _cancel_remote_stripe_subscription(stripe, subscription_id)
                _store_stripe_subscription(
                    conn,
                    subscription,
                    event_created_at=event_created_at,
                    provider_environment=_stripe_event_environment(event),
                    override_status="refunded",
                )

    # Insert only after every relevant update succeeds. A rollback leaves the
    # event retryable, while the composite key makes successful replays inert.
    return record_provider_event(
        conn,
        provider="stripe",
        external_event_id=event_id,
        event_type=event_type,
        external_event_created_at=event_created_at,
    )


def _cancel_remote_stripe_subscription(stripe: Any, subscription_id: str) -> Any:
    cancel = getattr(stripe.Subscription, "cancel", None)
    if callable(cancel):
        return cancel(subscription_id)
    return stripe.Subscription.delete(subscription_id)  # pragma: no cover


def cancel_stripe_for_account(conn: Any, account_id: str) -> bool:
    """Close both materialized and webhook-lagged Stripe billing work."""
    rows = [
        row
        for row in subscription_rows(conn, account_id)
        if str(row.get("provider") or "") == "stripe"
    ]
    provider_environment = billing_entitlement_environment()
    customer_id = provider_customer_id(
        conn,
        account_id,
        "stripe",
        provider_environment=provider_environment,
    )
    candidates = {
        str(row.get("external_subscription_id") or ""): str(
            row.get("status") or "unknown"
        ).strip().lower()
        for row in rows
        if str(row.get("external_subscription_id") or "").strip()
    }
    if not candidates and not customer_id:
        return False

    _stripe_configured_environment()
    stripe = _stripe_module()
    changed = False
    if customer_id:
        open_sessions = stripe.checkout.Session.list(
            customer=customer_id,
            status="open",
            limit=100,
        )
        for session in _obj_get(open_sessions, "data", []) or []:
            session_id = _id(session)
            if session_id:
                stripe.checkout.Session.expire(session_id)
                changed = True

        remote_subscriptions = stripe.Subscription.list(
            customer=customer_id,
            status="all",
            limit=100,
        )
        for subscription in _obj_get(remote_subscriptions, "data", []) or []:
            subscription_id = _id(subscription)
            if subscription_id:
                candidates[subscription_id] = str(
                    _obj_get(subscription, "status", "unknown") or "unknown"
                ).strip().lower()

    for subscription_id, status in candidates.items():
        if status in TERMINAL_SUBSCRIPTION_STATUSES:
            continue
        _cancel_remote_stripe_subscription(stripe, subscription_id)
        changed = True
    return changed


__all__ = [
    "BillingAccountNotFoundError",
    "BillingConfigurationError",
    "BillingVerificationError",
    "DuplicateSubscriptionError",
    "cancel_stripe_for_account",
    "construct_stripe_event",
    "create_stripe_checkout",
    "create_stripe_portal",
    "lock_billing_account",
    "process_stripe_event",
]
