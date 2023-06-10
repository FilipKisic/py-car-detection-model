# TAKE 3
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('trained_model.h5')

class_labels = ['bottle', 'butterfly', 'car']

image_path = 'cls.jpg'
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

target_size = (224, 224)
resized_image = cv2.resize(image, target_size)

preprocessed_image = resized_image.astype(np.float32) / 255.0
preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

predictions = model.predict(preprocessed_image)
class_index = np.argmax(predictions)
class_label = class_labels[class_index]
confidence = predictions[0][class_index]

bbox = predictions[0][:2] * np.array([target_size[1], target_size[0]])
bbox = bbox.astype(int)

bbox_bottom_right = bbox[:2] + predictions[0][2:4] * np.array([target_size[1], target_size[0]])
bbox_bottom_right = bbox_bottom_right.astype(int)

scale_factor = (image_width / target_size[1], image_height / target_size[0])
bbox = (bbox * scale_factor).astype(int)
bbox_bottom_right = (bbox_bottom_right * scale_factor).astype(int)

cv2.rectangle(image, (bbox[0], bbox[1]), (bbox_bottom_right[0], bbox_bottom_right[1]), (0, 255, 0), 2)
cv2.putText(image, f'{class_label}: {confidence:.2f}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0, 255, 0), 2)

print('Detected Class:', class_label)
print('Confidence:', confidence)

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
