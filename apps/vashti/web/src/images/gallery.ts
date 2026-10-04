import type { GalleryItem, ImageResult } from "./types";

export function appendGallery(items: GalleryItem[], page: GalleryItem[]) {
  const ids = new Set(items.map((item) => item.id));
  const added = page.filter((item) => {
    if (ids.has(item.id)) return false;
    ids.add(item.id);
    return true;
  });
  return [...items, ...added];
}

export async function galleryVariants(
  item: GalleryItem,
  fetchPage: (offset: number) => Promise<ImageResult[]>,
) {
  if (!item.group_id || item.variants <= 1) return [];
  const variants: ImageResult[] = [];
  for (let offset = 0; ; offset += 100) {
    const page = await fetchPage(offset);
    variants.push(...page.filter((value) => value.group_id === item.group_id));
    if (page.length < 100 || variants.length >= item.variants) break;
  }
  return variants.reverse();
}
