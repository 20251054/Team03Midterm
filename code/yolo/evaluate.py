from ultralytics import YOLO

model = YOLO("best.pt")
metrics = model.val(data="labeldFiled/data.yaml", split="test") # 경로 확인 필요
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
