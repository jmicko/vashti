use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
    time::SystemTime,
};

fn main() {
    println!("cargo:rerun-if-env-changed=VASHTI_SKIP_WEB_BUILD");
    println!("cargo:rerun-if-changed=web/index.html");
    println!("cargo:rerun-if-changed=web/package-lock.json");
    println!("cargo:rerun-if-changed=web/package.json");
    println!("cargo:rerun-if-changed=web/public");
    println!("cargo:rerun-if-changed=web/src");
    println!("cargo:rerun-if-changed=web/tsconfig.json");
    println!("cargo:rerun-if-changed=web/vite.config.ts");

    if env::var_os("VASHTI_SKIP_WEB_BUILD").is_some() {
        return;
    }

    match should_build_frontend() {
        Ok(true) => build_frontend(),
        Ok(false) => {}
        Err(error) => panic!("failed to inspect frontend build inputs: {error}"),
    }
}

fn should_build_frontend() -> io::Result<bool> {
    let dist_index = Path::new("web/dist/index.html");
    if !dist_index.exists() {
        return Ok(true);
    }

    let dist_modified = dist_index.metadata()?.modified()?;
    let newest_input = newest_modified_time(&[
        PathBuf::from("web/index.html"),
        PathBuf::from("web/package-lock.json"),
        PathBuf::from("web/package.json"),
        PathBuf::from("web/public"),
        PathBuf::from("web/src"),
        PathBuf::from("web/tsconfig.json"),
        PathBuf::from("web/vite.config.ts"),
    ])?;

    Ok(newest_input > dist_modified)
}

fn newest_modified_time(paths: &[PathBuf]) -> io::Result<SystemTime> {
    let mut newest = SystemTime::UNIX_EPOCH;

    for path in paths {
        if !path.exists() {
            continue;
        }

        newest = newest.max(path.metadata()?.modified()?);
        if path.is_dir() {
            newest = newest.max(newest_modified_time_in_dir(path)?);
        }
    }

    Ok(newest)
}

fn newest_modified_time_in_dir(path: &Path) -> io::Result<SystemTime> {
    let mut newest = SystemTime::UNIX_EPOCH;

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        newest = newest.max(path.metadata()?.modified()?);

        if path.is_dir() {
            newest = newest.max(newest_modified_time_in_dir(&path)?);
        }
    }

    Ok(newest)
}

fn build_frontend() {
    let npm = if cfg!(windows) { "npm.cmd" } else { "npm" };
    let status = Command::new(npm)
        .args(["run", "build"])
        .current_dir("web")
        .status()
        .unwrap_or_else(|error| panic!("failed to run npm in web/: {error}"));

    if !status.success() {
        panic!("frontend build failed with status {status}");
    }
}
