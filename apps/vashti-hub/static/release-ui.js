(() => {
  function formatBytes(value) {
    const units = ["B", "KiB", "MiB", "GiB"];
    let size = Number(value);
    let unit = 0;
    while (size >= 1024 && unit < units.length - 1) {
      size /= 1024;
      unit += 1;
    }
    return `${size.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
  }

  function titleCase(value) {
    return value
      .split(/[-_]/)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
  }

  function artifactPresentation(artifact) {
    const target = artifact.target.toLowerCase();
    if (target === "android-universal") {
      return { title: "Android app", detail: "Universal APK", linuxServer: false };
    }
    if (/^linux-(x86_64|aarch64|arm64)$/.test(target)) {
      const architecture = target.split("-").slice(1).join(" ");
      return { title: "Linux server", detail: `${architecture} archive`, linuxServer: true };
    }
    if (target.startsWith("windows-")) {
      return {
        title: "Windows desktop",
        detail: titleCase(target.replace(/^windows-/, "")),
        linuxServer: false,
      };
    }
    if (target.startsWith("macos-") || target.startsWith("darwin-")) {
      const detail = target.replace(/^(macos|darwin)-/, "");
      return {
        title: "macOS desktop",
        detail: detail === "aarch64" || detail === "arm64" ? "Apple silicon" : titleCase(detail),
        linuxServer: false,
      };
    }
    if (target.startsWith("linux-desktop-")) {
      return {
        title: "Linux desktop",
        detail: titleCase(target.replace(/^linux-desktop-/, "")),
        linuxServer: false,
      };
    }
    return {
      title: titleCase(artifact.target),
      detail: artifact.filename,
      linuxServer: false,
    };
  }

  function installCommandFor(release) {
    if (release.is_latest) {
      return "curl -fsSL https://vashti.chat/install.sh | sh";
    }
    return `curl -fsSL https://vashti.chat/install.sh | VASHTI_VERSION=${release.version} sh`;
  }

  async function copyText(text, button) {
    if (navigator.clipboard) {
      await navigator.clipboard.writeText(text);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      document.body.append(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    button.textContent = "Copied";
    setTimeout(() => {
      button.textContent = "Copy";
    }, 1800);
  }

  function createArtifactLink(artifact, showDownloadCounts) {
    const presentation = artifactPresentation(artifact);
    const link = document.createElement("a");
    link.className = "artifact-download";
    link.href = artifact.download_url;

    const copy = document.createElement("span");
    copy.className = "artifact-download-copy";
    const title = document.createElement("strong");
    title.textContent = presentation.title;
    const details = [presentation.detail, formatBytes(artifact.size_bytes)];
    if (showDownloadCounts) {
      const downloads = Number(artifact.downloads || 0);
      details.push(`${downloads.toLocaleString()} ${downloads === 1 ? "download" : "downloads"}`);
    }
    const meta = document.createElement("small");
    meta.textContent = details.join(" · ");
    copy.append(title, meta);

    const action = document.createElement("span");
    action.className = "artifact-download-action";
    action.textContent = "Download";
    link.append(copy, action);
    return { link, linuxServer: presentation.linuxServer };
  }

  function createReleaseDelivery(release, options = {}) {
    const showDownloadCounts = options.showDownloadCounts ?? false;
    const details = document.createElement("details");
    details.className = "release-delivery-details";

    const summary = document.createElement("summary");
    const summaryTitle = document.createElement("strong");
    summaryTitle.textContent = "Install & download";
    const summaryMeta = document.createElement("small");
    const artifactCount = release.artifacts.length;
    summaryMeta.textContent = `${artifactCount} ${artifactCount === 1 ? "build" : "builds"}`;
    summary.append(summaryTitle, summaryMeta);
    details.append(summary);

    const delivery = document.createElement("div");
    delivery.className = "release-delivery";
    const downloads = document.createElement("section");
    downloads.className = "release-delivery-group";
    const heading = document.createElement("h4");
    heading.textContent = "Direct downloads";
    const artifactList = document.createElement("div");
    artifactList.className = "artifact-download-list";
    let hasLinuxServer = false;
    for (const artifact of release.artifacts) {
      const rendered = createArtifactLink(artifact, showDownloadCounts);
      artifactList.append(rendered.link);
      hasLinuxServer ||= rendered.linuxServer;
    }
    downloads.append(heading, artifactList);
    delivery.append(downloads);

    if (hasLinuxServer) {
      const install = document.createElement("section");
      install.className = "release-delivery-group release-install";
      const installHeading = document.createElement("h4");
      installHeading.textContent = "Install Linux server";
      const description = document.createElement("p");
      description.className = "muted";
      description.textContent = release.is_latest
        ? "Installs the latest stable server and systemd service."
        : `Installs server ${release.version} and its systemd service.`;
      const command = document.createElement("div");
      command.className = "command compact";
      const code = document.createElement("code");
      code.textContent = installCommandFor(release);
      const copy = document.createElement("button");
      copy.type = "button";
      copy.textContent = "Copy";
      copy.addEventListener("click", () => copyText(code.textContent || "", copy));
      command.append(code, copy);
      install.append(installHeading, description, command);
      delivery.append(install);
    }

    details.append(delivery);
    return details;
  }

  window.VashtiReleaseUI = { createReleaseDelivery };
})();
