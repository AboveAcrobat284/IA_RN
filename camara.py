import cv2
import os

def capture_images_from_camera(output_dir, img_count, capture_interval=20):
    cap = cv2.VideoCapture(0)  # 0 es el índice de la cámara por defecto
    
    if not cap.isOpened():
        print("Error opening the camera")
        return img_count
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % capture_interval == 0:
            img_path = os.path.join(output_dir, f"img_{img_count:02d}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved image {img_path}")
            img_count += 1
        
        frame_count += 1

        # Mostrar el video en tiempo real
        cv2.imshow('Capturing Images', frame)
        
        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {img_count} images from the camera")
    return img_count

if __name__ == "__main__":
    output_directory = "Entrenamiento/SalsaRanch/"
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    img_count = 0
    capture_interval = 2  
    
    img_count = capture_images_from_camera(output_directory, img_count, capture_interval)
