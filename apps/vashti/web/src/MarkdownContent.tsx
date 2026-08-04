import {
  isValidElement,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import { Copy } from "lucide-react";
import "highlight.js/styles/github-dark.css";

const markdownComponents = {
  pre({ children }) {
    return <CodeBlock>{children}</CodeBlock>;
  }
} satisfies Components;

export function MarkdownContent({
  content,
  dimmedEmphasis = false
}: {
  content: string;
  dimmedEmphasis?: boolean;
}) {
  return (
    <div
      className={
        dimmedEmphasis
          ? "message-markdown message-markdown-dimmed-emphasis"
          : "message-markdown"
      }
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={markdownComponents}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function CodeBlock({ children }: { children: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const resetCopiedRef = useRef<number | null>(null);
  const codeText = useMemo(() => textFromReactNode(children).replace(/\n$/, ""), [children]);

  useEffect(() => {
    return () => {
      if (resetCopiedRef.current) {
        window.clearTimeout(resetCopiedRef.current);
      }
    };
  }, []);

  async function copyCode() {
    if (!codeText) {
      return;
    }

    await navigator.clipboard.writeText(codeText);
    setCopied(true);

    if (resetCopiedRef.current) {
      window.clearTimeout(resetCopiedRef.current);
    }

    resetCopiedRef.current = window.setTimeout(() => {
      setCopied(false);
      resetCopiedRef.current = null;
    }, 1400);
  }

  return (
    <div className="code-block">
      <button
        type="button"
        className="code-copy-button"
        title="Copy code"
        aria-label="Copy code"
        disabled={!codeText}
        onClick={() => void copyCode()}
      >
        <Copy />
        <span>{copied ? "Copied" : "Copy"}</span>
      </button>
      <pre>{children}</pre>
    </div>
  );
}

function textFromReactNode(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }

  if (Array.isArray(node)) {
    return node.map(textFromReactNode).join("");
  }

  if (isValidElement<{ children?: ReactNode }>(node)) {
    return textFromReactNode(node.props.children);
  }

  return "";
}
