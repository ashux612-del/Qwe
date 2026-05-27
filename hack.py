import cv2
import numpy as np

print("====================================")
print("     AI COIN TRACKER INITIALIZED    ")
print("====================================")

# 1. Image ko Download folder se load karte hain
image_path = '/sdcard/Download/carrom.jpg'
image = cv2.imread(image_path)

if image is None:
    print("❌ Error: 'carrom.jpg' file Download folder me nahi mili!")
    print("Kripya check karein ki naam sahi hai aur file Download folder me hai.")
else:
    print("✅ Screenshot successfully loaded!")
    
    # Image ko thoda chota karte hain processing tez karne ke liye
    h, w, _ = image.shape
    print(f"Original Image Size: {w}x{h}")
    
    # 2. BGR se Gray (Black & White) me convert karna circles dhoondhne ke liye
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Thoda blur karte hain taaki extra noise hat jaye
    blurred = cv2.medianBlur(gray, 5)
    
    # 3. HOUGH CIRCLES DETECTION (Gotiyan dhoondhna)
    # Aapke screenshot ke size ke hisaab se radius set kiya hai (approx 15 to 30 pixels)
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20, 
        param1=50, 
        param2=22, 
        minRadius=15, 
        maxRadius=35
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"🎯 Total Coins Detected: {len(circles[0])}\n")
        
        white_count = 0
        black_count = 0
        red_count = 0
        
        for idx, pt in enumerate(circles[0, :]):
            x, y, r = pt[0], pt[1], pt[2]
            
            # 4. COLOR DETECTION (Goti ke center ka color check karna)
            # OpenCV me color BGR format me hota hai
            roi = image[max(0, y-5):min(h, y+5), max(0, x-5):min(w, x+5)]
            avg_color = np.mean(roi, axis=(0, 1))
            b, g, r_val = avg_color[0], avg_color[1], avg_color[2]
            
            # Simple color logic based on RGB channels
            if r_val > 150 and b < 100 and g < 100:
                coin_color = "RED QUEEN ❤️"
                red_count += 1
            elif r_val > 140 and g > 140 and b > 140:
                coin_color = "WHITE COIN ⚪"
                white_count += 1
            else:
                coin_color = "BLACK COIN ⚫"
                black_count += 1
                
            print(f"Coin #{idx+1}: Position X={x}, Y={y} -> Type: {coin_color}")
            
        print("\n====================================")
        print(f"📊 SUMMARY:")
        print(f"White Coins: {white_count}")
        print(f"Black Coins: {black_count}")
        print(f"Red Queen: {red_count}")
    else:
        print("❌ Gotiyan detect nahi ho payin! Parameters tuning ki zaroorat hai.")

print("====================================")
