"use client";

import { type KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";

import { FadePresence } from "@/components/FadePresence";

export type CustomSelectOption<T extends string> = {
  value: T;
  label: string;
};

type CustomSelectProps<T extends string> = {
  label: string;
  value: T;
  options: ReadonlyArray<CustomSelectOption<T>>;
  onChange: (value: T) => void;
  name?: string;
  disabled?: boolean;
  className?: string;
  buttonClassName?: string;
  menuClassName?: string;
  showSelectedCheck?: boolean;
};

export function CustomSelect<T extends string>({
  label,
  value,
  options,
  onChange,
  name,
  disabled = false,
  className = "",
  buttonClassName = "",
  menuClassName = "",
  showSelectedCheck = false,
}: CustomSelectProps<T>) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const [opensAbove, setOpensAbove] = useState(false);
  const selectedIndex = Math.max(0, options.findIndex((option) => option.value === value));
  const selectedOption = options[selectedIndex];

  const focusOption = useCallback((index: number) => {
    const boundedIndex = Math.max(0, Math.min(options.length - 1, index));
    setActiveIndex(boundedIndex);
    window.requestAnimationFrame(() => optionRefs.current[boundedIndex]?.focus());
  }, [options.length]);

  const openMenu = useCallback(() => {
    if (disabled || options.length === 0) {
      return;
    }
    const triggerRect = triggerRef.current?.getBoundingClientRect();
    const estimatedMenuHeight = options.length * 40 + 12;
    setOpensAbove(Boolean(triggerRect && triggerRect.bottom + estimatedMenuHeight + 12 > window.innerHeight));
    setActiveIndex(selectedIndex);
    setOpen(true);
  }, [disabled, options.length, selectedIndex]);

  const selectOption = useCallback((option: CustomSelectOption<T>) => {
    onChange(option.value);
    setOpen(false);
    window.requestAnimationFrame(() => triggerRef.current?.focus());
  }, [onChange]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const frame = window.requestAnimationFrame(() => optionRefs.current[activeIndex]?.focus());
    const onPointerDown = (event: PointerEvent) => {
      if (event.target instanceof Node && !containerRef.current?.contains(event.target)) {
        setOpen(false);
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [activeIndex, open]);

  const onOptionKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      focusOption((index + 1) % options.length);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      focusOption((index - 1 + options.length) % options.length);
    } else if (event.key === "Home") {
      event.preventDefault();
      focusOption(0);
    } else if (event.key === "End") {
      event.preventDefault();
      focusOption(options.length - 1);
    } else if (event.key === "Tab") {
      setOpen(false);
    }
  };

  return (
    <div
      ref={containerRef}
      className={`relative ${className}`}
      data-custom-select="true"
      onKeyDownCapture={(event) => {
        if (!open || event.key !== "Escape") {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        setOpen(false);
        window.requestAnimationFrame(() => triggerRef.current?.focus());
      }}
    >
      {name ? <input type="hidden" name={name} value={value} /> : null}
      <button
        ref={triggerRef}
        type="button"
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={open}
        disabled={disabled}
        onClick={() => {
          if (open) {
            setOpen(false);
          } else {
            openMenu();
          }
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            openMenu();
          }
        }}
        className={`flex items-center justify-between gap-2 text-left transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white disabled:cursor-not-allowed disabled:opacity-45 ${buttonClassName}`}
      >
        <span className="min-w-0 truncate">{selectedOption?.label ?? value}</span>
        <svg viewBox="0 0 24 24" aria-hidden="true" className="h-3.5 w-3.5 shrink-0 fill-none stroke-current stroke-[1.5] text-white/52">
          <path d="m7.5 9.5 4.5 4.5 4.5-4.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      <FadePresence show={open}>
        {(visible) => (
          <div
            role="listbox"
            aria-label={label}
            aria-hidden={!visible}
            inert={!visible}
            className={`absolute left-0 z-50 min-w-full overflow-hidden rounded-2xl bg-[#202020] p-1.5 transition-opacity duration-300 motion-reduce:transition-none ${
              opensAbove ? "bottom-[calc(100%+8px)]" : "top-[calc(100%+8px)]"
            } ${visible ? "opacity-100" : "pointer-events-none opacity-0"} ${menuClassName}`}
          >
            {options.map((option, index) => {
              const selected = option.value === value;
              return (
                <button
                  key={option.value}
                  ref={(element) => {
                    optionRefs.current[index] = element;
                  }}
                  type="button"
                  role="option"
                  aria-selected={selected}
                  tabIndex={activeIndex === index ? 0 : -1}
                  onClick={() => selectOption(option)}
                  onKeyDown={(event) => onOptionKeyDown(event, index)}
                  className={`flex h-10 w-full items-center rounded-xl px-3 text-left text-sm transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-white ${
                    selected ? "bg-white/[0.11] text-white" : "text-white/72 hover:bg-white/[0.07] hover:text-white"
                  }`}
                >
                  <span className="min-w-0 truncate">{option.label}</span>
                  {showSelectedCheck && selected ? (
                    <span className="ml-auto shrink-0 pl-8" aria-hidden="true">
                      <svg
                        viewBox="0 0 24 24"
                        className="h-4 w-4 fill-none stroke-current stroke-[1.5]"
                      >
                        <path d="m5.5 12.5 4 4 9-9" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>
        )}
      </FadePresence>
    </div>
  );
}
