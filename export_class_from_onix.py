import onnx
import sys
import os

def extract_classes_from_onnx(model_path):
    # ONNX dosyasını yükleyin
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"ONNX dosyası yüklenirken hata oluştu: {e}")
        return

    # Model adı ve çıktı dosyası adı
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_file = f"{model_name}_classes.txt"

    # Metadata kontrolü (sınıf bilgisi burada olabilir)
    classes = None
    if model.metadata_props:
        for prop in model.metadata_props:
            if prop.key == "classes":  # Metadata'da 'classes' anahtarını arayın
                classes = prop.value.split(",")  # Sınıflar virgülle ayrılmış olabilir
                break

    if not classes:
        print("Metadata içinde sınıf bilgisi bulunamadı.")
        return

    # Sınıfları dosyaya kaydet
    try:
        with open(output_file, "w") as f:
            for cls in classes:
                f.write(cls.strip() + "\n")
        print(f"Sınıflar '{output_file}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"Çıktı dosyası oluşturulurken hata oluştu: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Kullanım: python extract_classes.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Belirtilen yol mevcut değil: {model_path}")
        sys.exit(1)

    extract_classes_from_onnx(model_path)

