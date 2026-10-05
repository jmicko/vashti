export function comparisonReference(result?: {
  asset_id: string;
  input_ids: string[];
}): string | undefined {
  const first = result?.input_ids[0];
  return first && first !== result?.asset_id ? first : undefined;
}

// Pointer holds are momentary; keyboard and assistive activation toggle.
export class ComparisonPress {
  private available = false;
  private held: string | null = null;
  private visible = false;
  private changed: (visible: boolean) => void;

  constructor(changed: (visible: boolean) => void) {
    this.changed = changed;
  }

  setAvailable(available: boolean) {
    this.available = available;
    if (!available) this.reset();
  }

  start(input: string) {
    if (!this.available || this.held) return;
    this.held = input;
    this.show(true);
  }

  end(input: string) {
    if (this.held !== input) return;
    this.reset();
  }

  activate() {
    if (this.available && !this.held) this.show(!this.visible);
  }

  reset() {
    this.held = null;
    this.show(false);
  }

  private show(visible: boolean) {
    if (this.visible === visible) return;
    this.visible = visible;
    this.changed(visible);
  }
}
