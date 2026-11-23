import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import {
  Compartment,
  EditorState,
  Transaction,
  type Extension,
} from "@codemirror/state";
import {
  drawSelection,
  EditorView,
  highlightActiveLineGutter,
  keymap,
  lineNumbers,
} from "@codemirror/view";
import {
  defaultKeymap,
  history,
  historyKeymap,
  indentWithTab,
  redo,
  undo,
} from "@codemirror/commands";
import {
  bracketMatching,
  defaultHighlightStyle,
  foldGutter,
  foldKeymap,
  indentOnInput,
  syntaxHighlighting,
} from "@codemirror/language";
import {
  highlightSelectionMatches,
  openSearchPanel,
  searchKeymap,
} from "@codemirror/search";
import { unifiedMergeView } from "@codemirror/merge";
import type { CanvasLanguage } from "./types";

export type CanvasEditorHandle = {
  focus: () => void;
  undo: () => void;
  redo: () => void;
  wrap: (before: string, after: string) => void;
  prefix: (prefix: string) => void;
  search: () => void;
};
type Props = {
  value: string;
  language: CanvasLanguage;
  readOnly?: boolean;
  original?: string;
  onChange: (value: string) => void;
  onSelection: (text: string) => void;
};
async function languageExtension(language: CanvasLanguage): Promise<Extension> {
  switch (language) {
    case "markdown":
      return (await import("@codemirror/lang-markdown")).markdown();
    case "javascript":
      return (await import("@codemirror/lang-javascript")).javascript({
        jsx: true,
      });
    case "typescript":
      return (await import("@codemirror/lang-javascript")).javascript({
        typescript: true,
        jsx: true,
      });
    case "python":
      return (await import("@codemirror/lang-python")).python();
    case "rust":
      return (await import("@codemirror/lang-rust")).rust();
    case "html":
      return (await import("@codemirror/lang-html")).html();
    case "css":
      return (await import("@codemirror/lang-css")).css();
    case "json":
      return (await import("@codemirror/lang-json")).json();
    case "sql":
      return (await import("@codemirror/lang-sql")).sql();
    default:
      return [];
  }
}
export const CanvasEditor = forwardRef<CanvasEditorHandle, Props>(
  function CanvasEditor(props, forwarded) {
    const parent = useRef<HTMLDivElement>(null),
      editor = useRef<EditorView | null>(null),
      callbacks = useRef(props);
    callbacks.current = props;
    const compartments = useRef({
      language: new Compartment(),
      readonly: new Compartment(),
      review: new Compartment(),
      gutter: new Compartment(),
      attributes: new Compartment(),
    });
    useImperativeHandle(
      forwarded,
      () => ({
        focus: () => editor.current?.focus(),
        undo: () => {
          if (editor.current && !callbacks.current.readOnly)
            undo(editor.current);
        },
        redo: () => {
          if (editor.current && !callbacks.current.readOnly)
            redo(editor.current);
        },
        wrap: (before, after) => {
          const view = editor.current;
          if (!view || callbacks.current.readOnly) return;
          const { from, to } = view.state.selection.main;
          const text = view.state.sliceDoc(from, to);
          view.dispatch({
            changes: { from, to, insert: before + text + after },
            selection: {
              anchor: from + before.length,
              head: to + before.length,
            },
            scrollIntoView: true,
          });
          view.focus();
        },
        prefix: (prefix) => {
          const view = editor.current;
          if (!view || callbacks.current.readOnly) return;
          const range = view.state.selection.main,
            first = view.state.doc.lineAt(range.from),
            last = view.state.doc.lineAt(range.to),
            changes = [];
          for (let n = first.number; n <= last.number; n++)
            changes.push({ from: view.state.doc.line(n).from, insert: prefix });
          view.dispatch({ changes, scrollIntoView: true });
          view.focus();
        },
        search: () => {
          const view = editor.current;
          if (view) openSearchPanel(view);
        },
      }),
      [],
    );
    useEffect(() => {
      if (!parent.current) return;
      const c = compartments.current;
      const view = new EditorView({
        parent: parent.current,
        state: EditorState.create({
          doc: callbacks.current.value,
          extensions: [
            history(),
            drawSelection(),
            highlightSelectionMatches(),
            indentOnInput(),
            bracketMatching(),
            syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
            keymap.of([
              ...defaultKeymap,
              ...historyKeymap,
              ...searchKeymap,
              ...foldKeymap,
              indentWithTab,
            ]),
            EditorView.lineWrapping,
            c.language.of([]),
            c.readonly.of([]),
            c.review.of([]),
            c.gutter.of([]),
            c.attributes.of(
              EditorView.contentAttributes.of({
                "aria-label": "Canvas content",
                spellcheck: "true",
              }),
            ),
            EditorView.updateListener.of((update) => {
              if (
                update.docChanged &&
                !update.transactions.every(
                  (t) =>
                    t.annotation(Transaction.userEvent) === "canvas.external",
                )
              )
                callbacks.current.onChange(update.state.doc.toString());
              if (update.selectionSet) {
                const { from, to } = update.state.selection.main;
                callbacks.current.onSelection(update.state.sliceDoc(from, to));
              }
            }),
            EditorView.theme({
              "&": {
                height: "100%",
                fontSize: "15px",
                backgroundColor: "var(--panel)",
                color: "var(--text)",
              },
              ".cm-scroller": {
                overflow: "auto",
                fontFamily: "inherit",
                lineHeight: "1.65",
              },
              ".cm-content": {
                padding: "20px 24px 40px",
                caretColor: "var(--text)",
              },
              ".cm-line": { padding: "0" },
              ".cm-gutters": {
                backgroundColor: "var(--panel-strong)",
                color: "var(--muted)",
                border: "0",
              },
              ".cm-activeLineGutter": { backgroundColor: "var(--hover-muted)" },
              ".cm-cursor": { borderLeftColor: "var(--text)" },
              "&.cm-focused": { outline: "none" },
              "&.cm-focused .cm-selectionBackground,.cm-selectionBackground,::selection":
                {
                  backgroundColor:
                    "color-mix(in srgb,var(--accent) 22%,transparent) !important",
                },
              ".cm-panels": {
                backgroundColor: "var(--panel-strong)",
                color: "var(--text)",
              },
              ".cm-searchMatch": { backgroundColor: "#75631866" },
              ".cm-insertedLine": { backgroundColor: "#36714b30" },
              ".cm-deletedChunk": { backgroundColor: "#94414130" },
            }),
          ],
        }),
      });
      editor.current = view;
      return () => {
        editor.current = null;
        view.destroy();
      };
    }, []);
    useEffect(() => {
      const view = editor.current;
      if (view && view.state.doc.toString() !== props.value)
        view.dispatch({
          changes: { from: 0, to: view.state.doc.length, insert: props.value },
          annotations: [
            Transaction.addToHistory.of(false),
            Transaction.userEvent.of("canvas.external"),
          ],
        });
    }, [props.value]);
    useEffect(() => {
      const view = editor.current;
      if (!view) return;
      let cancelled = false;
      void languageExtension(props.language)
        .catch(() => [])
        .then((extension) => {
          if (!cancelled)
            view.dispatch({
              effects: compartments.current.language.reconfigure(extension),
            });
        });
      const prose = props.language === "markdown" || props.language === "text";
      view.dispatch({
        effects: [
          compartments.current.gutter.reconfigure(
            prose
              ? []
              : [lineNumbers(), foldGutter(), highlightActiveLineGutter()],
          ),
          compartments.current.attributes.reconfigure(
            EditorView.contentAttributes.of({
              "aria-label": "Canvas content",
              spellcheck: String(prose),
            }),
          ),
        ],
      });
      return () => {
        cancelled = true;
      };
    }, [props.language]);
    useEffect(() => {
      editor.current?.dispatch({
        effects: compartments.current.readonly.reconfigure([
          EditorState.readOnly.of(!!props.readOnly),
          EditorView.editable.of(!props.readOnly),
        ]),
      });
    }, [props.readOnly]);
    useEffect(() => {
      editor.current?.dispatch({
        effects: compartments.current.review.reconfigure(
          props.original === undefined
            ? []
            : unifiedMergeView({
                original: props.original,
                mergeControls: false,
                highlightChanges: true,
                diffConfig: { scanLimit: 2000, timeout: 100 },
              }),
        ),
      });
    }, [props.original]);
    return (
      <div
        className={`canvas-codemirror ${props.language === "markdown" || props.language === "text" ? "canvas-prose" : "canvas-code"}`}
        ref={parent}
      />
    );
  },
);
