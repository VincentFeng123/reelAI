"use client";

export function safeStorageSetItem(storage: Storage | null | undefined, key: string, value: string): boolean {
  if (!storage) {
    return false;
  }
  try {
    storage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

export function safeStorageRemoveItem(storage: Storage | null | undefined, key: string): boolean {
  if (!storage) {
    return false;
  }
  try {
    storage.removeItem(key);
    return true;
  } catch {
    return false;
  }
}
