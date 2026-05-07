(() => {
  const header = document.querySelector("[data-topbar]");
  const toggle = document.querySelector("[data-nav-toggle]");
  const nav = document.querySelector("[data-site-nav]");

  if (!header || !toggle || !nav) return;

  const setOpen = (open) => {
    header.dataset.navOpen = open ? "true" : "false";
    toggle.setAttribute("aria-expanded", String(open));
  };

  toggle.addEventListener("click", (event) => {
    event.stopPropagation();
    setOpen(header.dataset.navOpen !== "true");
  });

  nav.addEventListener("click", (event) => {
    if (event.target instanceof Element && event.target.closest("a")) {
      setOpen(false);
    }
  });

  document.addEventListener("click", (event) => {
    if (header.dataset.navOpen === "true" && !header.contains(event.target)) {
      setOpen(false);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setOpen(false);
  });
})();
