# User Setup Choices

Vashti separates an effective setting value from the fact that a person has
explicitly chosen it. This lets useful defaults remain enabled without silently
introducing new model access after an update.

## Behavior

After authentication, `/api/auth/session` reports whether the account has any
unresolved setup choices. The frontend shows a blocking setup screen when it
does. Every unresolved choice is shown together, so choices from several missed
releases accumulate instead of replacing one another.

Recommended values are selected initially. A person may accept them together or
change individual toggles before continuing. All choices remain editable from
their normal Personal settings pages afterward. Signing out is always available
from setup.

## Decision Ledger

`user_setting_decisions` records:

* the user and stable setting key
* the last explicitly chosen value
* whether the decision came from setup, a regular settings page, or a migration
* the original and latest decision timestamps

The feature settings tables remain the runtime source of truth. The ledger only
answers whether the person has made an explicit choice. Regular settings saves
must update both in the same transaction.

## Adding A Choice

1. Add a stable key and definition to `user_setup::service::CHOICES`.
2. Add its normal feature-setting storage and update path.
3. Record the key whenever the normal settings UI saves that field.
4. Do not backfill a decision for existing users when the choice is genuinely
   new; it will appear after their next login.
5. Add migration decisions only when adopting a pre-existing setting that users
   should not be asked to choose again.
6. Add dependency validation when one choice requires another permission.

New accounts receive no automatic decision rows, so every registered choice is
shown. Adding another registry entry later therefore works for both new and
long-absent users without a separate onboarding version number.

## Current Choices

The initial registry covers model permissions for server Notes, Memories, and
past-chat search. Private device data is excluded because its preferences live
in encrypted device storage rather than the server account.
