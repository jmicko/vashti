import {
  isValidElement,
  memo,
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
import { DocumentAttachment } from "./attachments";
import { importedAttachmentId } from "./imports/fileLinks";
import { pdfCitation } from "./pdf/types";
import type { AttachmentInfo } from "./types";
import "highlight.js/styles/github-dark.css";

const markdownComponents = {
  pre({ children }) {
    return <CodeBlock>{children}</CodeBlock>;
  }
} satisfies Components;
const deferredImageComponents = {
  ...markdownComponents,
  img({ src, alt }) { return <a href={src}>{alt || "Image"}</a>; }
} satisfies Components;

export const MarkdownContent = memo(function MarkdownContent({
  content,
  dimmedEmphasis = false,
  deferImages = false,
  importedAttachments,
  pdfAttachments
}: {
  content: string;
  dimmedEmphasis?: boolean;
  deferImages?: boolean;
  importedAttachments?: AttachmentInfo[];
  pdfAttachments?: AttachmentInfo[];
}) {
  const components = useMemo<Components>(() => {
    const base=deferImages?deferredImageComponents:markdownComponents;
    const find=(href:string)=> {
      const id=importedAttachmentId(href);
      return id===undefined?undefined:importedAttachments?.find(file=>file.id===id)??null;
    };
    const render: Components = {...base,
      a({node:_node,href,children,...props}){
        const citation = pdfCitation(href ?? "");
        if (citation !== undefined) {
          const pdf = citation && pdfAttachments?.find(file => file.id === citation.id);
          return pdf ? <DocumentAttachment attachment={pdf} inlineLabel={children} pdfPage={citation.page}/>
            : <span title="PDF reference is not available on this branch">{children}</span>;
        }
        const file=find(href??"");
        return file===undefined?<a {...props} href={href}>{children}</a>:file
          ?<DocumentAttachment attachment={file} inlineLabel={children??file.original_filename}/>
          :<span title="File not available. Resume the ZIP import if it was interrupted.">{children} (file unavailable)</span>;
      },
    };
    if (deferImages || importedAttachments) render.img = ({src,alt}) => {
      const file=find(typeof src==="string"?src:"");
      return file===undefined?<a href={src}>{alt||"Image"}</a>:file
        ?<DocumentAttachment attachment={file} inlineLabel={alt||file.original_filename}/>
        :<span>{alt||"Image"} (file unavailable)</span>;
    };
    return render;
  },[deferImages,importedAttachments,pdfAttachments]);
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
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});

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
