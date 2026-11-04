import { Check, Minus } from "lucide-react";
import type { AccessStatus } from "./effectiveAccess";

export function EffectiveAccess({
  status,
  pending = false,
}: {
  status: AccessStatus[];
  pending?: boolean;
}) {
  return (
    <section className="effective-access" aria-label="Effective model access">
      <h3>{pending ? "Access after save" : "Effective access"}</h3>
      <dl>
        {status.map(({ action, allowed, reason }) => (
          <div key={action}>
            <dt>{action}</dt>
            <dd>
              {allowed ? (
                <Check aria-hidden="true" />
              ) : (
                <Minus aria-hidden="true" />
              )}
              {reason}
            </dd>
          </div>
        ))}
      </dl>
      <small>Conversation tool controls also apply.</small>
    </section>
  );
}
