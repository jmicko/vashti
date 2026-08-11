import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import {
  Bold,
  CheckSquare,
  Code2,
  Heading2,
  Italic,
  Link,
  List,
  ListOrdered,
  Minus,
  MoreHorizontal,
  Quote
} from "lucide-react";

type MarkdownToolbarProps = {
  textareaRef: RefObject<HTMLTextAreaElement | null>;
  value: string;
  onChange: (value: string) => void;
};

export function MarkdownToolbar({ textareaRef, value, onChange }: MarkdownToolbarProps) {
  const [isMoreOpen, setIsMoreOpen] = useState(false);
  const moreRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isMoreOpen) {
      return;
    }

    function close(event: PointerEvent) {
      if (event.target instanceof Node && moreRef.current?.contains(event.target)) {
        return;
      }
      setIsMoreOpen(false);
    }

    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  }, [isMoreOpen]);

  function apply(transform: MarkdownTransform) {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const result = transform(value, start, end);
    onChange(result.value);
    setIsMoreOpen(false);
    window.requestAnimationFrame(() => {
      textarea.focus();
      textarea.setSelectionRange(result.selectionStart, result.selectionEnd);
    });
  }

  return (
    <div className="notes-markdown-toolbar" role="toolbar" aria-label="Markdown formatting">
      <FormatButton label="Bold" onClick={() => apply(wrapSelection("**", "**", "bold text"))}>
        <Bold />
      </FormatButton>
      <FormatButton label="Italic" onClick={() => apply(wrapSelection("*", "*", "italic text"))}>
        <Italic />
      </FormatButton>
      <FormatButton label="Heading" onClick={() => apply(prefixLines("## "))}>
        <Heading2 />
      </FormatButton>
      <FormatButton label="Bulleted list" onClick={() => apply(prefixLines("- "))}>
        <List />
      </FormatButton>
      <FormatButton label="Checklist" onClick={() => apply(prefixLines("- [ ] "))}>
        <CheckSquare />
      </FormatButton>
      <div className="notes-format-more" ref={moreRef}>
        <FormatButton
          label="More formatting"
          isPressed={isMoreOpen}
          onClick={() => setIsMoreOpen((open) => !open)}
        >
          <MoreHorizontal />
        </FormatButton>
        {isMoreOpen && (
          <div className="notes-format-menu">
            <button type="button" className="menu-item" onClick={() => apply(prefixLines("1. "))}>
              <ListOrdered />
              <span>Numbered list</span>
            </button>
            <button type="button" className="menu-item" onClick={() => apply(prefixLines("> "))}>
              <Quote />
              <span>Quote</span>
            </button>
            <button
              type="button"
              className="menu-item"
              onClick={() => apply(wrapSelection("`", "`", "code"))}
            >
              <Code2 />
              <span>Inline code</span>
            </button>
            <button
              type="button"
              className="menu-item"
              onClick={() => apply(wrapSelection("[", "](https://)", "link text"))}
            >
              <Link />
              <span>Link</span>
            </button>
            <button type="button" className="menu-item" onClick={() => apply(insertBlock("\n---\n"))}>
              <Minus />
              <span>Divider</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function FormatButton({
  children,
  label,
  isPressed,
  onClick
}: {
  children: ReactNode;
  label: string;
  isPressed?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="icon-button notes-format-button"
      aria-label={label}
      title={label}
      aria-pressed={isPressed}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

type MarkdownTransform = (
  value: string,
  start: number,
  end: number
) => { value: string; selectionStart: number; selectionEnd: number };

function wrapSelection(before: string, after: string, placeholder: string): MarkdownTransform {
  return (value, start, end) => {
    const selected = value.slice(start, end) || placeholder;
    const replacement = `${before}${selected}${after}`;
    return {
      value: value.slice(0, start) + replacement + value.slice(end),
      selectionStart: start + before.length,
      selectionEnd: start + before.length + selected.length
    };
  };
}

function prefixLines(prefix: string): MarkdownTransform {
  return (value, start, end) => {
    const lineStart = value.lastIndexOf("\n", Math.max(0, start - 1)) + 1;
    const nextLine = value.indexOf("\n", end);
    const lineEnd = nextLine === -1 ? value.length : nextLine;
    const selected = value.slice(lineStart, lineEnd);
    const replacement = selected
      .split("\n")
      .map((line) => `${prefix}${line}`)
      .join("\n");
    return {
      value: value.slice(0, lineStart) + replacement + value.slice(lineEnd),
      selectionStart: start + prefix.length,
      selectionEnd: end + prefix.length * selected.split("\n").length
    };
  };
}

function insertBlock(block: string): MarkdownTransform {
  return (value, start, end) => ({
    value: value.slice(0, start) + block + value.slice(end),
    selectionStart: start + block.length,
    selectionEnd: start + block.length
  });
}
