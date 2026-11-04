# Image workflow browser regression fixture

Run the normal Vite dev server and open `/tests/images.html`. This entry is not
included in production builds. All application requests are intercepted by the
fixture; unexpected requests fail rather than reaching a real inference host.
Device gallery data uses its own encrypted test vault, never a signed-in user's
vault. The images and prompts are synthetic.

Check at 390x844 and desktop sizes:

- Page through the 550-entry dated gallery while delayed previews arrive. The
  fixture reuses 60 synthetic image assets to keep test storage bounded; entries
  still have distinct library IDs. Tile dimensions and
  scroll position must stay fixed. The diagnostic counter should record preview
  requests, not original-image requests, until a full image is opened.
- Open image 4. Version arrows switch its two retries; swiping, image arrows and
  keyboard arrows move to separate library entries.
- Tap the photo to toggle overlays; they also fade after inactivity. Double-click
  or Zoom in, then drag to pan without changing photos. Fit and swipe again;
  consecutive swipes must not wait for the previous settling animation.
- More opens secondary image actions. Back closes More, then the photo, keeping
  the gallery's scroll position. Keyboard-focused controls must not disappear.
- Open image 50 and advance. Image 51 must open without losing the grid
  position. Close during loading and verify the viewer does not reopen.
- Switch to Device gallery and repeat navigation. It must use encrypted local
  assets without incrementing the network counters.
- In the server gallery, More > Only this conversation on image 4 narrows the
  library to that stack. Clear the filter to return to the whole library.
- Use Fail next originals before opening an uncached photo. It rejects three
  original requests (including neighbors); Retry image must recover the selected
  photo without a page reload. Thumbnail requests are unaffected.
- Open Image preferences, change account and server fields, and save each scope
  separately. Diagnostic write counters must increment only for the saved scope.
  Dirty Close must offer Cancel/Discard above the preferences sheet; Escape
  dismisses only the confirmation. These writes never reach real settings.
- Open History and immediately search Saved drawing 64 before the slower first
  page returns. Only that matching conversation should remain. Check an empty
  search, rename/refiltering, and removal with these disposable records in both
  Server gallery and Device gallery scopes.
- Open Edit flow, then its reference-count button. Move reference 3 left, then
  reference 2 left, and close the reference sheet. Send. The submitted list in
  the diagnostic output must be reference-3, reference-1, reference-2.
- Send repeatedly in Original references mode. All turns must remain in one
  conversation and use the same references. Leave for Gallery and reopen Edit
  flow to check restored ancestry.
- Switch Edit source to Latest image. The next completed output becomes the
  following edit's sole reference. Add two gallery references, remove the first,
  reorder, and Send. The thread must remain connected, and the following Send
  must use that edited output.
- Tap reference thumbnails to inspect them, then close the viewer. Neither the
  reference order nor the unsent prompt should change.
- Open image inference settings from the header. Settings must appear separately
  from the composer; closing them preserves the prompt/reference summary.
- At 390px width, reference sheets, model menus, and Gallery View must have no
  horizontal overflow. Check 844x390 and desktop without resetting photo zoom
  merely because controls appear/disappear.

Run `node_modules/.bin/tsc -p tests/tsconfig.json` to type-check the fixture.
The fixture keeps simulated flow records in memory; reloading resets its flow.
Automated persistence/ownership tests live in `src/images/deviceRepository.test.mjs`
and `apps/vashti/src/images/tests.rs`.
