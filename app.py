from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from src import config
from src.gradcam import generate_gradcam
from src.predict import load_trained_model, predict_image

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
CACHED_MODEL = None
MODEL_LOAD_ERROR = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_cached_model():
    global CACHED_MODEL, MODEL_LOAD_ERROR

    if CACHED_MODEL is not None:
        return CACHED_MODEL

    if MODEL_LOAD_ERROR is not None:
        raise MODEL_LOAD_ERROR

    try:
        CACHED_MODEL = load_trained_model()
        return CACHED_MODEL
    except Exception as exc:
        MODEL_LOAD_ERROR = exc
        raise


def save_uploaded_file(file_storage) -> tuple[Path, str]:
    original_name = secure_filename(file_storage.filename or "")
    if not original_name:
        raise ValueError("Tên file không hợp lệ.")

    unique_name = f"{uuid4().hex}_{original_name}"
    save_path = Path(config.UPLOAD_DIR) / unique_name
    file_storage.save(save_path)
    return save_path, unique_name


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "result": None,
        "error": None,
        "uploaded_image_url": None,
        "uploaded_filename": None,
        "gradcam_url": None,
        "gradcam_warning": None,
    }

    if request.method == "POST":
        uploaded = request.files.get("image")

        if uploaded is None:
            context["error"] = "Vui lòng chọn một ảnh để tải lên."
            return render_template("index.html", **context)

        if uploaded.filename == "":
            context["error"] = "Tên file trống. Vui lòng chọn file ảnh hợp lệ."
            return render_template("index.html", **context)

        if not allowed_file(uploaded.filename):
            context["error"] = "Định dạng không hỗ trợ. Chỉ chấp nhận JPG, JPEG hoặc PNG."
            return render_template("index.html", **context)

        try:
            saved_path, unique_name = save_uploaded_file(uploaded)
            model = get_cached_model()
            prediction = predict_image(saved_path, model=model)

            context["result"] = prediction
            context["uploaded_filename"] = uploaded.filename
            context["uploaded_image_url"] = url_for("uploaded_file", filename=unique_name)

            try:
                gradcam_filename = generate_gradcam(saved_path, model=model)
                context["gradcam_url"] = url_for("output_file", filename=gradcam_filename)
            except Exception as gradcam_exc:
                app.logger.warning("Grad-CAM generation failed: %s", gradcam_exc)
                context["gradcam_warning"] = (
                    "Không thể tạo ảnh Grad-CAM cho ảnh này. "
                    "Kết quả dự đoán chính vẫn hợp lệ."
                )

        except FileNotFoundError:
            context["error"] = (
                "Chưa tìm thấy model đã train. Vui lòng chạy lệnh: python -m src.train"
            )
        except ValueError:
            context["error"] = (
                "Không thể xử lý ảnh đã tải lên. Vui lòng dùng ảnh JPG/PNG rõ ràng và thử lại."
            )
        except RuntimeError:
            context["error"] = "Hệ thống dự đoán gặp sự cố. Vui lòng thử lại sau."
        except Exception:
            context["error"] = "Đã xảy ra lỗi không mong muốn. Vui lòng thử lại."

    return render_template("index.html", **context)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    safe_name = secure_filename(filename)
    if not safe_name or safe_name != filename:
        return ("Not found", 404)
    return send_from_directory(config.UPLOAD_DIR, safe_name)


@app.route("/outputs/<path:filename>")
def output_file(filename: str):
    safe_name = secure_filename(filename)
    if not safe_name or safe_name != filename:
        return ("Not found", 404)
    return send_from_directory(config.OUTPUT_DIR, safe_name)


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
