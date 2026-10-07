// X-ray preview, drag & drop, and a loading overlay while the models run.
const dz = document.getElementById("dropzone"), input = document.getElementById("xray"), img = document.getElementById("preview");
if (dz && input) {
  const show = f => { if (!f || !f.type.startsWith("image/")) return; img.src = URL.createObjectURL(f); dz.classList.add("has-img"); };
  input.addEventListener("change", e => show(e.target.files[0]));
  ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, () => dz.classList.remove("drag")));
  dz.addEventListener("drop", e => { e.preventDefault(); input.files = e.dataTransfer.files; show(input.files[0]); });
}
const form = document.getElementById("screen-form");
if (form) form.addEventListener("submit", () => {
  const o = document.createElement("div"); o.className = "loading on";
  o.innerHTML = '<div><div class="spinner"></div><h3 style="justify-content:center">Running the screening…</h3></div>';
  document.body.appendChild(o);
});
