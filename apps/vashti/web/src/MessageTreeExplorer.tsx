import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronsDown,
  LocateFixed,
  Minus,
  Plus,
  X
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent
} from "react";
import { activePathMessages, revisionsForMessage } from "./chatMessages";
import { MarkdownContent } from "./MarkdownContent";
import type { ChatMessage, ChatMessageRevision } from "./types";

const NODE_WIDTH = 224;
const NODE_HEIGHT = 118;
const HORIZONTAL_GAP = 38;
const VERTICAL_GAP = 78;
const CANVAS_PADDING = 72;
const MIN_SCALE = 0.24;
const MAX_SCALE = 1.8;

type TreeNode = {
  id: string;
  message: ChatMessage;
  revision: ChatMessageRevision;
  revisionNumber: number;
};

type MessageTreeGroup = {
  id: string;
  message: ChatMessage;
  nodes: TreeNode[];
  parentMessageId: string | null;
  childMessageIds: string[];
};

type PositionedTreeNode = TreeNode & {
  x: number;
  y: number;
};

type TreeEdge = {
  id: string;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  active: boolean;
};

type TreeJunction = {
  id: string;
  messageId: string;
  x: number;
  y: number;
  active: boolean;
};

type TreeLayout = {
  nodes: PositionedTreeNode[];
  edges: TreeEdge[];
  junctions: TreeJunction[];
  width: number;
  height: number;
};

type ViewTransform = {
  x: number;
  y: number;
  scale: number;
};

type PointerPosition = {
  x: number;
  y: number;
};

export function MessageTreeExplorer({
  activeRootMessageId,
  messages,
  onClose,
  onOpenBranch
}: {
  activeRootMessageId: string | null;
  messages: ChatMessage[];
  onClose: () => void;
  onOpenBranch: (messageId: string, revisionId: string) => void;
}) {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const pointersRef = useRef(new Map<number, PointerPosition>());
  const pointerStartsRef = useRef(new Map<number, PointerPosition>());
  const viewRef = useRef<ViewTransform>({ x: 0, y: 0, scale: 1 });
  const hasPositionedRef = useRef(false);
  const [view, setView] = useState<ViewTransform>(viewRef.current);
  const [collapsedMessageIds, setCollapsedMessageIds] = useState<Set<string>>(
    () => new Set()
  );
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null);

  const tree = useMemo(() => buildMessageTree(messages), [messages]);
  const activeBranch = useMemo(() => {
    const path = activePathMessages(messages, activeRootMessageId);
    const nodeIds = path.flatMap((message) =>
      message.active_revision_id
        ? [messageTreeNodeId(message.id, message.active_revision_id)]
        : []
    );
    return {
      nodeIds: new Set(nodeIds),
      messageIds: new Set(path.map((message) => message.id)),
      leafNodeId: nodeIds[nodeIds.length - 1] ?? null
    };
  }, [activeRootMessageId, messages]);
  const layout = useMemo(
    () =>
      layoutMessageTree(
        tree,
        collapsedMessageIds,
        activeBranch.nodeIds,
        activeBranch.messageIds
      ),
    [activeBranch.messageIds, activeBranch.nodeIds, collapsedMessageIds, tree]
  );
  const focusedNode =
    (focusedNodeId && tree.nodesById.get(focusedNodeId)) || null;

  const applyView = useCallback((nextView: ViewTransform) => {
    viewRef.current = nextView;
    setView(nextView);
  }, []);

  const fitTree = useCallback(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }

    const rect = viewport.getBoundingClientRect();
    const availableWidth = Math.max(1, rect.width - 48);
    const availableHeight = Math.max(1, rect.height - 96);
    const scale = clamp(
      Math.min(1, availableWidth / layout.width, availableHeight / layout.height),
      MIN_SCALE,
      MAX_SCALE
    );
    applyView({
      x: (rect.width - layout.width * scale) / 2,
      y: Math.max(72, (rect.height - layout.height * scale) / 2),
      scale
    });
  }, [applyView, layout.height, layout.width]);

  const positionInitialView = useCallback(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }

    const rect = viewport.getBoundingClientRect();
    const availableWidth = Math.max(1, rect.width - 48);
    const availableHeight = Math.max(1, rect.height - 96);
    const fitScale = clamp(
      Math.min(1, availableWidth / layout.width, availableHeight / layout.height),
      MIN_SCALE,
      MAX_SCALE
    );
    if (fitScale >= 0.68 || !activeBranch.leafNodeId) {
      fitTree();
      return;
    }

    const target = layout.nodes.find((node) => node.id === activeBranch.leafNodeId);
    if (!target) {
      fitTree();
      return;
    }

    const scale = clamp(
      Math.min(
        1,
        availableWidth / (NODE_WIDTH + 96),
        availableHeight / (NODE_HEIGHT + 160)
      ),
      0.72,
      1
    );
    applyView({
      x: rect.width / 2 - (target.x + NODE_WIDTH / 2) * scale,
      y: rect.height / 2 - (target.y + NODE_HEIGHT / 2) * scale,
      scale
    });
  }, [
    activeBranch.leafNodeId,
    applyView,
    fitTree,
    layout.height,
    layout.nodes,
    layout.width
  ]);

  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }

    const observer = new ResizeObserver(() => {
      if (!hasPositionedRef.current) {
        hasPositionedRef.current = true;
        positionInitialView();
      }
    });
    observer.observe(viewport);
    return () => observer.disconnect();
  }, [positionInitialView]);

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key !== "Escape") {
        return;
      }
      if (focusedNodeId) {
        setFocusedNodeId(null);
      } else {
        onClose();
      }
    }

    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [focusedNodeId, onClose]);

  function zoomAt(point: PointerPosition, nextScale: number) {
    const current = viewRef.current;
    const scale = clamp(nextScale, MIN_SCALE, MAX_SCALE);
    const worldX = (point.x - current.x) / current.scale;
    const worldY = (point.y - current.y) / current.scale;
    applyView({
      x: point.x - worldX * scale,
      y: point.y - worldY * scale,
      scale
    });
  }

  function zoomFromCenter(multiplier: number) {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }
    const rect = viewport.getBoundingClientRect();
    zoomAt(
      { x: rect.width / 2, y: rect.height / 2 },
      viewRef.current.scale * multiplier
    );
  }

  function handleWheel(event: ReactWheelEvent<HTMLDivElement>) {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    zoomAt(
      { x: event.clientX - rect.left, y: event.clientY - rect.top },
      viewRef.current.scale * Math.exp(-event.deltaY * 0.0015)
    );
  }

  function handlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    const target = event.target;
    if (target instanceof Element && target.closest("button, .message-tree-focus")) {
      return;
    }

    event.currentTarget.setPointerCapture(event.pointerId);
    const rect = event.currentTarget.getBoundingClientRect();
    const position = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
    pointersRef.current.set(event.pointerId, position);
    pointerStartsRef.current.set(event.pointerId, position);
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const previousPosition = pointersRef.current.get(event.pointerId);
    if (!previousPosition) {
      return;
    }

    const rect = event.currentTarget.getBoundingClientRect();
    const nextPosition = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
    const previousPointers = new Map(pointersRef.current);
    pointersRef.current.set(event.pointerId, nextPosition);

    if (pointersRef.current.size === 1) {
      const current = viewRef.current;
      applyView({
        ...current,
        x: current.x + nextPosition.x - previousPosition.x,
        y: current.y + nextPosition.y - previousPosition.y
      });
      return;
    }

    if (pointersRef.current.size !== 2) {
      return;
    }

    const pointerIds = [...pointersRef.current.keys()];
    const previousA = previousPointers.get(pointerIds[0]);
    const previousB = previousPointers.get(pointerIds[1]);
    const nextA = pointersRef.current.get(pointerIds[0]);
    const nextB = pointersRef.current.get(pointerIds[1]);
    if (!previousA || !previousB || !nextA || !nextB) {
      return;
    }

    const previousDistance = distance(previousA, previousB);
    const nextDistance = distance(nextA, nextB);
    if (previousDistance <= 0) {
      return;
    }

    const previousCenter = midpoint(previousA, previousB);
    const nextCenter = midpoint(nextA, nextB);
    const current = viewRef.current;
    const nextScale = clamp(
      current.scale * (nextDistance / previousDistance),
      MIN_SCALE,
      MAX_SCALE
    );
    const worldX = (previousCenter.x - current.x) / current.scale;
    const worldY = (previousCenter.y - current.y) / current.scale;
    applyView({
      x: nextCenter.x - worldX * nextScale,
      y: nextCenter.y - worldY * nextScale,
      scale: nextScale
    });
  }

  function releasePointer(event: ReactPointerEvent<HTMLDivElement>) {
    pointersRef.current.delete(event.pointerId);
    pointerStartsRef.current.delete(event.pointerId);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function toggleSubtree(messageId: string) {
    setCollapsedMessageIds((current) => {
      const next = new Set(current);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  }

  return (
    <section className="message-tree-explorer" aria-label="Message tree explorer">
      <header className="message-tree-toolbar">
        <div>
          <strong>Message Tree</strong>
          <small>
            {tree.nodes.length} {tree.nodes.length === 1 ? "version" : "versions"}
          </small>
        </div>
        <div className="message-tree-toolbar-actions">
          {collapsedMessageIds.size > 0 && (
            <button
              type="button"
              className="secondary-button message-tree-expand-all"
              onClick={() => setCollapsedMessageIds(new Set())}
            >
              <ChevronsDown />
              <span>Expand All</span>
            </button>
          )}
          <button
            type="button"
            className="icon-button"
            aria-label="Zoom out"
            title="Zoom out"
            onClick={() => zoomFromCenter(0.82)}
          >
            <Minus />
          </button>
          <button
            type="button"
            className="icon-button"
            aria-label="Zoom in"
            title="Zoom in"
            onClick={() => zoomFromCenter(1.22)}
          >
            <Plus />
          </button>
          <button
            type="button"
            className="icon-button"
            aria-label="Fit tree to view"
            title="Fit tree to view"
            onClick={fitTree}
          >
            <LocateFixed />
          </button>
          <button
            type="button"
            className="icon-button"
            aria-label="Close message tree"
            title="Close message tree"
            onClick={onClose}
          >
            <X />
          </button>
        </div>
      </header>
      <div className="message-tree-stage">
        <div
          ref={viewportRef}
          className="message-tree-viewport"
          onPointerCancel={releasePointer}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={releasePointer}
          onWheel={handleWheel}
        >
          {layout.nodes.length === 0 ? (
            <div className="message-tree-empty">No messages yet.</div>
          ) : (
            <div
              className="message-tree-canvas"
              style={{
                width: layout.width,
                height: layout.height,
                transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`
              }}
            >
              <svg
                className="message-tree-edges"
                width={layout.width}
                height={layout.height}
                aria-hidden="true"
              >
                {layout.edges.map((edge) => {
                  const bendY = edge.startY + (edge.endY - edge.startY) / 2;
                  return (
                    <path
                      key={edge.id}
                      className={
                        edge.active
                          ? "message-tree-edge message-tree-edge-active"
                          : "message-tree-edge"
                      }
                      d={`M ${edge.startX} ${edge.startY} C ${edge.startX} ${bendY}, ${edge.endX} ${bendY}, ${edge.endX} ${edge.endY}`}
                    />
                  );
                })}
                {layout.junctions.map((junction) => (
                  <circle
                    key={junction.id}
                    className={
                      junction.active
                        ? "message-tree-junction message-tree-junction-active"
                        : "message-tree-junction"
                    }
                    cx={junction.x}
                    cy={junction.y}
                    r="5"
                  />
                ))}
              </svg>
              {layout.nodes.map((node) => {
                const group = tree.groupsById.get(node.message.id);
                const isCollapsed = collapsedMessageIds.has(node.message.id);
                const childCount = group?.childMessageIds.length ?? 0;
                const hasChildren = childCount > 0;
                const collapseNodeId =
                  group?.nodes.find(
                    (groupNode) =>
                      groupNode.revision.id === group.message.active_revision_id
                  )?.id ??
                  group?.nodes[group.nodes.length - 1]?.id ??
                  null;
                return (
                  <article
                    key={node.id}
                    className={
                      activeBranch.nodeIds.has(node.id)
                        ? "message-tree-node message-tree-node-active"
                        : "message-tree-node"
                    }
                    style={{ transform: `translate(${node.x}px, ${node.y}px)` }}
                  >
                    <button
                      type="button"
                      className="message-tree-node-open"
                      onClick={() => setFocusedNodeId(node.id)}
                    >
                      <header>
                        <span>{treeNodeLabel(node)}</span>
                        {node.message.revisions.length > 1 && (
                          <small>edit {node.revisionNumber}</small>
                        )}
                      </header>
                      <p>{treeNodePreview(node)}</p>
                      <footer>
                        <span>{node.message.status}</span>
                      </footer>
                    </button>
                    {hasChildren && node.id === collapseNodeId && (
                      <button
                        type="button"
                        className="message-tree-collapse"
                        aria-label={
                          isCollapsed ? "Expand subtree" : "Collapse subtree"
                        }
                        title={isCollapsed ? "Expand subtree" : "Collapse subtree"}
                        onClick={() => toggleSubtree(node.message.id)}
                      >
                        {isCollapsed ? <ChevronRight /> : <ChevronDown />}
                        <span>{childCount}</span>
                      </button>
                    )}
                  </article>
                );
              })}
            </div>
          )}
        </div>
        {focusedNode && (
          <section className="message-tree-focus" aria-label="Focused message">
            <article
              className={`message-tree-focus-card message-bubble message-bubble-${focusedNode.message.role}`}
            >
              <header>
                <button
                  type="button"
                  className="icon-button"
                  aria-label="Back to message tree"
                  title="Back to message tree"
                  onClick={() => setFocusedNodeId(null)}
                >
                  <ChevronLeft />
                </button>
                <div>
                  <strong>{treeNodeLabel(focusedNode)}</strong>
                  <small>
                    Edit {focusedNode.revisionNumber} of{" "}
                    {Math.max(1, focusedNode.message.revisions.length)}
                  </small>
                </div>
              </header>
              <div className="message-tree-focus-content">
                {focusedNode.revision.thinking_text && (
                  <details>
                    <summary>Thinking</summary>
                    <MarkdownContent content={focusedNode.revision.thinking_text} />
                  </details>
                )}
                {focusedNode.revision.content_text ? (
                  <MarkdownContent content={focusedNode.revision.content_text} />
                ) : (
                  <p className="muted">
                    {focusedNode.message.is_deleted
                      ? "This message was deleted."
                      : "This message has no visible content."}
                  </p>
                )}
              </div>
              <footer>
                <button
                  type="button"
                  onClick={() =>
                    onOpenBranch(focusedNode.message.id, focusedNode.revision.id)
                  }
                >
                  Open This Branch
                </button>
              </footer>
            </article>
          </section>
        )}
      </div>
    </section>
  );
}

export function buildMessageTree(messages: ChatMessage[]) {
  const nodes: TreeNode[] = [];
  const groups: MessageTreeGroup[] = [];
  const messagesById = new Map(messages.map((message) => [message.id, message]));

  for (const message of messages) {
    const revisions = revisionsForMessage(message);
    const groupNodes = revisions.map((revision, index) => {
      const node = {
        id: messageTreeNodeId(message.id, revision.id),
        message,
        revision,
        revisionNumber: index + 1
      };
      nodes.push(node);
      return node;
    });
    groups.push({
      id: message.id,
      message,
      nodes: groupNodes,
      parentMessageId: message.parent_message_id,
      childMessageIds: []
    });
  }

  const nodesById = new Map(nodes.map((node) => [node.id, node]));
  const groupsById = new Map(groups.map((group) => [group.id, group]));
  for (const group of groups) {
    if (!group.parentMessageId) {
      continue;
    }

    const parentGroup = groupsById.get(group.parentMessageId);
    if (!parentGroup) {
      continue;
    }
    parentGroup.childMessageIds.push(group.id);
  }

  for (const group of groups) {
    group.childMessageIds.sort((leftId, rightId) =>
      compareMessages(messagesById.get(leftId), messagesById.get(rightId))
    );
  }

  const roots = groups
    .filter(
      (group) =>
        !group.parentMessageId || !groupsById.has(group.parentMessageId)
    )
    .sort((left, right) => compareMessages(left.message, right.message));

  return { groups, groupsById, nodes, nodesById, roots };
}

export function layoutMessageTree(
  tree: ReturnType<typeof buildMessageTree>,
  collapsedMessageIds: Set<string>,
  activeNodeIds: Set<string>,
  activeMessageIds: Set<string>
): TreeLayout {
  const subtreeWidths = new Map<string, number>();
  const positionedNodes: PositionedTreeNode[] = [];
  const groupCenters = new Map<string, { x: number; y: number }>();
  const visibleMessageIds = new Set<string>();
  const laidOutMessageIds = new Set<string>();
  const hiddenMessageIds = new Set<string>();

  function hideDescendants(
    messageId: string,
    ancestors = new Set<string>([messageId])
  ) {
    const group = tree.groupsById.get(messageId);
    if (!group) {
      return;
    }

    for (const childId of group.childMessageIds) {
      if (ancestors.has(childId)) {
        continue;
      }
      hiddenMessageIds.add(childId);
      const nextAncestors = new Set(ancestors);
      nextAncestors.add(childId);
      hideDescendants(childId, nextAncestors);
    }
  }

  for (const messageId of collapsedMessageIds) {
    hideDescendants(messageId);
  }

  function ownWidth(group: MessageTreeGroup) {
    return Math.max(
      NODE_WIDTH,
      group.nodes.length * NODE_WIDTH +
        Math.max(0, group.nodes.length - 1) * HORIZONTAL_GAP
    );
  }

  function measure(group: MessageTreeGroup, ancestors = new Set<string>()): number {
    const cached = subtreeWidths.get(group.id);
    if (cached !== undefined) {
      return cached;
    }
    if (ancestors.has(group.id)) {
      return ownWidth(group);
    }

    const nextAncestors = new Set(ancestors);
    nextAncestors.add(group.id);
    const children = collapsedMessageIds.has(group.id)
      ? []
      : group.childMessageIds.flatMap((messageId) => {
          const child = tree.groupsById.get(messageId);
          return child && !nextAncestors.has(child.id) ? [child] : [];
        });
    const childrenWidth =
      children.reduce((total, child) => total + measure(child, nextAncestors), 0) +
      Math.max(0, children.length - 1) * HORIZONTAL_GAP;
    const width = Math.max(ownWidth(group), childrenWidth);
    subtreeWidths.set(group.id, width);
    return width;
  }

  function place(
    group: MessageTreeGroup,
    left: number,
    depth: number,
    ancestors = new Set<string>()
  ) {
    if (ancestors.has(group.id) || laidOutMessageIds.has(group.id)) {
      return;
    }

    const nextAncestors = new Set(ancestors);
    nextAncestors.add(group.id);
    const width = measure(group);
    const centerX = left + width / 2;
    const y = CANVAS_PADDING + depth * (NODE_HEIGHT + VERTICAL_GAP);
    const groupWidth = ownWidth(group);
    const groupLeft = centerX - groupWidth / 2;

    group.nodes.forEach((node, index) => {
      positionedNodes.push({
        ...node,
        x: groupLeft + index * (NODE_WIDTH + HORIZONTAL_GAP),
        y
      });
    });
    groupCenters.set(group.id, { x: centerX, y });
    visibleMessageIds.add(group.id);
    laidOutMessageIds.add(group.id);

    if (collapsedMessageIds.has(group.id)) {
      return;
    }

    const children = group.childMessageIds.flatMap((messageId) => {
      const child = tree.groupsById.get(messageId);
      return child && !nextAncestors.has(child.id) ? [child] : [];
    });
    const childrenWidth =
      children.reduce((total, child) => total + measure(child, nextAncestors), 0) +
      Math.max(0, children.length - 1) * HORIZONTAL_GAP;
    let childLeft = left + (width - childrenWidth) / 2;
    for (const child of children) {
      place(child, childLeft, depth + 1, nextAncestors);
      childLeft += measure(child, nextAncestors) + HORIZONTAL_GAP;
    }
  }

  let rootLeft = CANVAS_PADDING;
  for (const root of tree.roots) {
    if (hiddenMessageIds.has(root.id)) {
      continue;
    }
    place(root, rootLeft, 0);
    rootLeft += measure(root) + HORIZONTAL_GAP * 1.5;
  }
  for (const group of tree.groups) {
    if (laidOutMessageIds.has(group.id) || hiddenMessageIds.has(group.id)) {
      continue;
    }
    place(group, rootLeft, 0);
    rootLeft += measure(group) + HORIZONTAL_GAP * 1.5;
  }

  const visibleNodesByMessageId = new Map<string, PositionedTreeNode[]>();
  for (const node of positionedNodes) {
    const groupNodes = visibleNodesByMessageId.get(node.message.id) ?? [];
    groupNodes.push(node);
    visibleNodesByMessageId.set(node.message.id, groupNodes);
  }

  const edges: TreeEdge[] = [];
  const junctions: TreeJunction[] = [];
  for (const messageId of visibleMessageIds) {
    const group = tree.groupsById.get(messageId);
    const center = groupCenters.get(messageId);
    const groupNodes = visibleNodesByMessageId.get(messageId) ?? [];
    const visibleChildren = (group?.childMessageIds ?? []).flatMap((childId) =>
      visibleMessageIds.has(childId) ? [childId] : []
    );
    if (!group || !center || groupNodes.length === 0 || visibleChildren.length === 0) {
      continue;
    }

    const junctionY = center.y + NODE_HEIGHT + VERTICAL_GAP * 0.42;
    junctions.push({
      id: `junction:${messageId}`,
      messageId,
      x: center.x,
      y: junctionY,
      active: activeMessageIds.has(messageId)
    });
    for (const node of groupNodes) {
      edges.push({
        id: `merge:${node.id}`,
        startX: node.x + NODE_WIDTH / 2,
        startY: node.y + NODE_HEIGHT,
        endX: center.x,
        endY: junctionY,
        active: activeNodeIds.has(node.id)
      });
    }
    for (const childMessageId of visibleChildren) {
      const childNodes = visibleNodesByMessageId.get(childMessageId) ?? [];
      for (const childNode of childNodes) {
        edges.push({
          id: `branch:${messageId}->${childNode.id}`,
          startX: center.x,
          startY: junctionY,
          endX: childNode.x + NODE_WIDTH / 2,
          endY: childNode.y,
          active:
            activeMessageIds.has(messageId) &&
            activeNodeIds.has(childNode.id)
        });
      }
    }
  }

  const nodes = positionedNodes;
  const maxX = nodes.reduce(
    (maximum, node) => Math.max(maximum, node.x + NODE_WIDTH),
    0
  );
  const maxY = nodes.reduce(
    (maximum, node) => Math.max(maximum, node.y + NODE_HEIGHT),
    0
  );

  return {
    nodes,
    edges,
    junctions,
    width: Math.max(1, maxX + CANVAS_PADDING),
    height: nodes.length > 0 ? maxY + CANVAS_PADDING : 1
  };
}

function messageTreeNodeId(messageId: string, revisionId: string) {
  return `${messageId}:${revisionId}`;
}

function treeNodeLabel(node: Pick<TreeNode, "message">) {
  if (node.message.role === "user") {
    return "You";
  }
  return (
    node.message.persona_name_snapshot ??
    node.message.model_name ??
    node.message.role
  );
}

function treeNodePreview(node: TreeNode) {
  const content =
    node.revision.content_text.trim() ||
    node.revision.thinking_text.trim() ||
    (node.message.is_deleted ? "Deleted message" : "Empty message");
  return content.length > 190 ? `${content.slice(0, 187).trimEnd()}...` : content;
}

function compareMessages(left?: ChatMessage, right?: ChatMessage) {
  if (!left || !right) {
    return 0;
  }
  return (
    left.created_at - right.created_at ||
    left.id.localeCompare(right.id)
  );
}

function midpoint(left: PointerPosition, right: PointerPosition) {
  return {
    x: (left.x + right.x) / 2,
    y: (left.y + right.y) / 2
  };
}

function distance(left: PointerPosition, right: PointerPosition) {
  return Math.hypot(left.x - right.x, left.y - right.y);
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}
