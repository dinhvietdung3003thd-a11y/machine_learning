(function () {
  const input = document.getElementById("image");
  const preview = document.getElementById("image-preview");
  const filenameText = document.getElementById("selected-file");

  if (!input || !preview || !filenameText) {
    return;
  }

  const allowed = ["jpg", "jpeg", "png"];

  input.addEventListener("change", function () {
    const file = input.files && input.files[0];
    if (!file) {
      preview.classList.add("hidden");
      preview.removeAttribute("src");
      filenameText.textContent = "Chưa chọn file.";
      return;
    }

    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (!allowed.includes(ext)) {
      preview.classList.add("hidden");
      preview.removeAttribute("src");
      filenameText.textContent = "File không hợp lệ (chỉ JPG/JPEG/PNG).";
      return;
    }

    filenameText.textContent = `Đã chọn: ${file.name}`;
    preview.src = URL.createObjectURL(file);
    preview.classList.remove("hidden");
  });
})();
