// Light mode is not supported. Force dark theme regardless of user / system
// preference. Furo reads from document.body.dataset.theme and localStorage.
(function () {
  try {
    localStorage.setItem("theme", "dark");
  } catch (e) {}

  function applyDark() {
    if (document.body) {
      document.body.dataset.theme = "dark";
    }
    if (document.documentElement) {
      document.documentElement.dataset.theme = "dark";
    }
  }

  applyDark();
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", applyDark);
  }
})();
