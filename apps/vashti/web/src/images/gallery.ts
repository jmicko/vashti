import type { GalleryItem, ImageResult } from "./types";

export function galleryDays(items: GalleryItem[], now = new Date()) {
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  const days: Array<{ key: string; label: string; items: GalleryItem[] }> = [];
  for (const item of items) {
    const date = new Date(item.created_at * 1000);
    const key = date.toDateString();
    let day = days[days.length - 1];
    if (day?.key !== key) {
      day = {
        key,
        items: [],
        label:
          key === today.toDateString()
            ? "Today"
            : key === yesterday.toDateString()
              ? "Yesterday"
              : date.toLocaleDateString(undefined, {
                  month: "long",
                  day: "numeric",
                  year: "numeric",
                }),
      };
      days.push(day);
    }
    day.items.push(item);
  }
  return days;
}

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
  if (!item.group_id) return [];
  const variants: ImageResult[] = [];
  for (let offset = 0; ; offset += 100) {
    const page = await fetchPage(offset);
    variants.push(...page.filter((value) => value.group_id === item.group_id));
    if (page.length < 100 || variants.length >= item.variants) break;
  }
  return variants.reverse();
}
