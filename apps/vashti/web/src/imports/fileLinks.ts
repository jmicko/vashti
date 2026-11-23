const prefix = "/.vashti/imported-file/";
const source = /^asset-[a-f0-9]{64}$/;
const attachment = /^(?:chatgpt-asset-[a-f0-9]{64}|[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})$/;

export function importedSourceLink(id: string) {
  if (!source.test(id)) throw new Error("Invalid imported asset identifier");
  return prefix + id;
}

export function importedAttachmentId(href: string) {
  if (!href.startsWith(prefix)) return undefined;
  const id = href.slice(prefix.length);
  return attachment.test(id) ? id : null;
}

export function bindImportedFileLinks(content: string, ids: ReadonlyMap<string, string>) {
  if (!ids.size) return content;
  return content.replace(/\/\.vashti\/imported-file\/(asset-[a-f0-9]{64})(?=[\s)>]|$)/g,
    (original, id: string) => ids.has(id) ? prefix + ids.get(id) : original);
}
