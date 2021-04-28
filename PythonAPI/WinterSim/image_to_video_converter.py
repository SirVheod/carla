import cv2
import os

def images_to_video(images, video_path, fps):
    if images:
        frame = images[0]
        height, width, layers = frame.shape

        if '.' not in video_path:
            video_path += '.avi'
            
        video = cv2.VideoWriter(video_path, 0, fps, (width,height))
        for image in images:
            video.write(image)
            
        cv2.destroyAllWindows()
        video.release()
        print("done")

def image_folder_to_video(image_folder, video_path, fps):
    images = [cv2.imread(os.path.join(image_folder, img))
              for img in os.listdir(image_folder)
              if img.endswith(".jpg")]
    images_to_video(images, video_path, fps)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--image_folder', required=True, help='Location of the image folder')
    parser.add_argument('--video_path', required=True, help='Path to the output video')
    parser.add_argument('--fps', type=int, default=30)

    args = parser.parse_args()
    image_folder_to_video(args.image_folder, args.video_path, args.fps)


#C:/Carla/carla/PythonAPI/WinterSim/images
#python image_to_video_converter.py --image_folder=C:/Carla/carla/PythonAPI/WinterSim/images --video_path=C:/Carla/carla/PythonAPI/WinterSim/images --fps=15