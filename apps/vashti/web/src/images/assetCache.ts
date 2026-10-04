export type CachedImage = {
  url: string;
  width: number;
  height: number;
  bytes: number;
};
type Entry = {
  promise: Promise<CachedImage>;
  image?: CachedImage;
  users: number;
};

export class AssetCache {
  private entries = new Map<string, Entry>();
  private closed = false;
  private running = 0;
  private waiting: (() => void)[] = [];
  private load: (id: string) => Promise<CachedImage>;
  private limit: number;
  private byteLimit: number;
  private concurrency: number;
  private revoke: (url: string) => void;
  private validate: () => void;

  constructor(
    load: (id: string) => Promise<CachedImage>,
    limit = 10,
    byteLimit = 64 * 1024 * 1024,
    concurrency = 3,
    revoke = (url: string) => URL.revokeObjectURL(url),
    validate = () => {},
  ) {
    this.load = load;
    this.limit = limit;
    this.byteLimit = byteLimit;
    this.concurrency = concurrency;
    this.revoke = revoke;
    this.validate = validate;
  }

  peek(id: string) {
    return this.entries.get(id)?.image;
  }

  private async decode(id: string) {
    if (this.running >= this.concurrency)
      await new Promise<void>((resolve) => this.waiting.push(resolve));
    else this.running++;
    try {
      if (this.closed) throw new Error("Image workspace closed");
      const image = await this.load(id);
      if (this.closed) {
        this.revoke(image.url);
        throw new Error("Image workspace closed");
      }
      return image;
    } finally {
      const next = this.waiting.shift();
      if (next) next();
      else this.running--;
    }
  }

  acquire(id: string) {
    this.validate();
    if (this.closed) throw new Error("Image workspace closed");
    let entry = this.entries.get(id);
    if (!entry) {
      const created: Entry = {
        users: 0,
        promise: Promise.resolve(null as unknown as CachedImage),
      };
      created.promise = this.decode(id)
        .then((image) => {
          created.image = image;
          this.trim();
          return image;
        })
        .catch((error) => {
          if (this.entries.get(id) === created) this.entries.delete(id);
          throw error;
        });
      entry = created;
      this.entries.set(id, entry);
    }
    this.entries.delete(id);
    this.entries.set(id, entry);
    entry.users++;
    const acquired = entry;
    let released = false;
    return {
      image: entry.image,
      promise: entry.promise,
      release: () => {
        if (released) return;
        released = true;
        acquired.users--;
        this.trim();
      },
    };
  }

  private trim() {
    let bytes = [...this.entries.values()].reduce(
      (total, entry) => total + (entry.image?.bytes ?? 0),
      0,
    );
    for (const [id, entry] of this.entries) {
      if (this.entries.size <= this.limit && bytes <= this.byteLimit) break;
      if (!entry.users && entry.image) {
        this.revoke(entry.image.url);
        bytes -= entry.image.bytes;
        this.entries.delete(id);
      }
    }
  }

  dispose() {
    this.closed = true;
    for (const entry of this.entries.values())
      if (entry.image) this.revoke(entry.image.url);
    this.entries.clear();
  }
}

export function thumbnailCrop(width: number, height: number) {
  const side = Math.min(width, height);
  return { x: (width - side) / 2, y: (height - side) / 2, side };
}
