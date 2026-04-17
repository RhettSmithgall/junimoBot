import cv2
import numpy as np

def outline_objects(scene_path, template_path, threshold=0.8):
    # Load images
    scene = scene_path
    template = cv2.imread(template_path)
    
    if scene is None or template is None:
        raise ValueError("Check image paths; one or both could not be loaded.")

    h, w = template.shape[:2]
    
    # Convert to grayscale
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGRA2BGR)
    temp_gray = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)

    # Perform matching
    res = cv2.matchTemplate(scene_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
    
    # Find locations above threshold
    loc = np.where(res >= threshold)
    
    
    rects = []
    for pt in zip(*loc[::-1]):
        rects.append([int(pt[0]), int(pt[1]), int(w), int(h)])
        rects.append([int(pt[0]), int(pt[1]), int(w), int(h)]) # Duplicate for groupRectangles
    
    # groupRectangles merges overlapping boxes
    rects, _ = cv2.groupRectangles(rects, 1, 0.2)

    # Draw the outlines
    for (x, y, w, h) in rects:
        cv2.rectangle(scene, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(scene, template_path, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    return scene

# examople usage:
# screenshot = sct.grab(gameScreen)
# screenshot = np.array(screenshot)
# screenshot = detectimage.outline_objects(screenshot, 'player.png', threshold=0.6)