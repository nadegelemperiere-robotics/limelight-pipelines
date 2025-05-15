# -------------------------------------------------------
# Copyright (c) [2025] Nadege LEMPERIERE
# All rights reserved
# -------------------------------------------------------
# Pipeline post computing sample orientation for into
# the deep challenge
# -------------------------------------------------------
# Nadège LEMPERIERE, @2nd April 2025
# Latest revision: 2nd April 2025
# -------------------------------------------------------

# System includes
from math import atan, degrees, cos, sin, tan, sqrt

# OpenCV includes
import cv2
import numpy as np

class OrientationComputation :
    """ Post processes samples to compute orientation"""

    def __init__(self) :
        """ Constructor with default configuration """

        self.__hue_yellow_min = 15
        self.__hue_yellow_max = 80
        self.__hue_blue_min = 80
        self.__hue_blue_max = 120
        self.__hue_red_min1 = 0
        self.__hue_red_max1 = 10
        self.__hue_red_min2 = 170
        self.__hue_red_max2 = 180

        self.__index = -1

    def configure(self, conf) :
        """ 
        Configuration function 
        Parameters :
            conf (dict)   :  Configuration data to use
        """

        if "hue-yellow-min" in conf : self.__hue_yellow_min = conf["hue-yellow-min"]
        if "hue-yellow-max" in conf : self.__hue_yellow_max = conf["hue-yellow-max"]
        if "hue-blue-min" in conf : self.__hue_blue_min = conf["hue-blue-min"]
        if "hue-blue-max" in conf : self.__hue_blue_max = conf["hue-blue-max"]
        if "hue-red-min1" in conf : self.__hue_red_min1 = conf["hue-red-min1"]
        if "hue-red-max1" in conf : self.__hue_red_max1 = conf["hue-red-max1"]
        if "hue-red-min2" in conf : self.__hue_red_min2 = conf["hue-red-min2"]
        if "hue-red-max2" in conf : self.__hue_red_max2 = conf["hue-red-max2"]

        self.__index = -1
    
    def __draw_sample(image, color, angle, rect, xmin, xmax, ymin, ymax) :
        """
        Sample drawing function
        Parameters :
            image (array)           : The image on which to draw
            color (int)             : The color of the sample we want to draw
            angle (float)           : The angle of the sample we want to draw
            rect (array)            : The sample surrounding oriented rectangle from which orientation was derived
            xmin, xmax, ymin, ymax  : The bounding box of the sample in image
        """

        if rect is not None :
            box = cv2.boxPoints(rect)        # 4 corner points
            box = np.intp(box)

            rgb = (255,255,255)
            if color == 0 : rgb = (0,0,255)
            elif color == 1 : rgb = (255,0,0)
            elif color == 2 : rgb = (0,255,255)

            (text_w, text_h), baseline = cv2.getTextSize(f"{int(angle)}", cv2.FONT_HERSHEY_DUPLEX, 0.3, 1)
            cv2.drawContours(image, [box], 0, rgb, 1)
            cv2.rectangle(image,(int(rect[0][0] - 2 ), int(rect[0][1] - text_h - 5)), (int(rect[0][0] + text_w + 2), int(rect[0][1])),rgb,cv2.FILLED)
            cv2.putText(image, f"{int(angle)}", (int(rect[0][0]), int(rect[0][1]) - 2),cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 0), 1)
            cv2.rectangle(image,(int(xmin ), int(ymin)), (int(xmax), int(ymax)),rgb,1)
            
    def __format_inputs(self,inputs) :
        """
        Transform the llrobot data into a more understandable structure
        Parameters :
            inputs (array)  : The llrobot data as an array of flatten samples features
        Returns :
            - A table of all samples as dict
        """
         
        
        result = []

        if inputs is not None and (len(inputs) - 1) % 7 == 0 and len(inputs) > 0: 
            # Manage possible pipeline invalid inputs : None, or [0,0,0,0,0,0,0] or []

            # The first llrobot data is the identifier of the limelight image from which the 
            # Samples are coming. 
            # Since we shall keep sending the same llrobot data during a long time to make sure the
            # Camera got them, we have to make sure we don't reinitialize the tracking each time 
            # We receive the same data. 
            self.__index = inputs[0]

            for i_sample in range(int((len(inputs) - 1) / 7)) :

                formatted = {}

                formatted['index'] = inputs[7 * i_sample + 1]

                formatted['color'] = inputs[7 * i_sample + 6]

                formatted['x'] = inputs[7 * i_sample + 2]
                formatted['y'] = inputs[7 * i_sample + 3]
                formatted['sx'] = inputs[7 * i_sample + 4]
                formatted['sy'] = inputs[7 * i_sample + 5]
                formatted['xmin'] = formatted['x'] - formatted['sx'] / 2
                formatted['xmax'] = formatted['x'] + formatted['sx'] / 2
                formatted['ymin'] = formatted['y'] - formatted['sy'] / 2
                formatted['ymax'] = formatted['y'] + formatted['sy'] / 2

                formatted['area'] = inputs[7 * i_sample + 7]

                result.append(formatted)

        return result
    
    def __estimate_rect_size_from_bounding_box(xmin, ymin, xmax, ymax, angle_rad):
        
        a = xmax - xmin
        b = ymax - ymin

        A = np.array([
            [abs(cos(angle_rad)), abs(sin(angle_rad))],
            [abs(sin(angle_rad)), abs(cos(angle_rad))]
        ])
        b_vec = np.array([a, b])

        # Solve for [w, h]
        w_h = np.linalg.solve(A, b_vec)
        return w_h[0], w_h[1]
    
    def __estimate_min_distance_along_rectangle(box, distance) : 
        
        distances = []

        for i in range(4):
            p1 = box[i]
            p2 = box[(i + 1) % 4]
            # Get points along the edge
            num_points = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))  # length in pixels
            if num_points == 0:
                continue
            x_vals = np.linspace(p1[0], p2[0], num_points)
            y_vals = np.linspace(p1[1], p2[1], num_points)

            # Clamp coordinates to stay within image bounds
            x_vals = np.clip(x_vals, 0, distance.shape[1] - 1)
            y_vals = np.clip(y_vals, 0, distance.shape[0] - 1)

            # Sample distance values using bilinear interpolation
            for x, y in zip(x_vals, y_vals):
                ix, iy = int(x), int(y)
                distances.append(float(distance[iy, ix]))

        if not distances:
            return 0.0

        return float(np.mean(distances))

    def __process_area(self, image, yellow, blue, red, distance, x,y,sx,sy,color,index, save) :
        """
        Process image around a given sample
        Parameters :
            image (Mat) : Raw image
            yellow, blue, red (Mat) : color masks
            distance (Mat) : Map of the distance from a contour
            x,y,sx,sy : Sample center and bounding box size
            color : Sample color
            index : Sample index
            save : Whether intermediate results shall be saved
        Returns :
            - The estimated sample angle, the rectangle it came from and the intermediate results
        """

        debug = {}

        ratio = 0
        if sx != 0 :
            ratio = sy * 1.0 / sx
        if ratio != 2.33 :   
            tan_angle = (ratio * 2.33 - 1) / (2.33 - ratio)
            candidate1 = atan(tan_angle) 
            while candidate1 < 0 : candidate1 += 3.1415927
            while candidate1 >= 3.1415927 : candidate1 -= 3.1415927
            candidate2 = 3.1415927 - candidate1
        else :
            candidate1 = candidate2 = 3.1415927 / 2

        width, height = OrientationComputation.__estimate_rect_size_from_bounding_box(x - sx/2,y - sy/2, x + sx/2, y + sy/2,candidate1)

        rotated1 = (x,y),(width, height),degrees(candidate1)
        rotated2 = (x,y),(width, height),degrees(candidate2)
        box1 = cv2.boxPoints(rotated1)  # 4x2 float32
        box1 = box1.astype(np.int32)
        box2 = cv2.boxPoints(rotated2)  # 4x2 float32
        box2 = box2.astype(np.int32)

        mask1 = np.zeros(image.shape[:2], dtype=np.uint8)
        box1 = cv2.boxPoints(rotated1)
        box1 = np.intp(box1)
        cv2.drawContours(mask1, [box1], 0, 255, thickness=cv2.FILLED)
        
        mask2 = np.zeros(image.shape[:2], dtype=np.uint8)
        box2 = cv2.boxPoints(rotated2) 
        box2 = np.intp(box2) 
        cv2.drawContours(mask2, [box2], 0, 255, thickness=cv2.FILLED)

        if color == 0 :
            combined_mask1 = cv2.bitwise_and(mask1, red)
            combined_mask2 = cv2.bitwise_and(mask2, red)
        elif color == 1 :
            combined_mask1 = cv2.bitwise_and(mask1, blue)
            combined_mask2 = cv2.bitwise_and(mask2, blue)
        elif color == 2 :
            combined_mask1 = cv2.bitwise_and(mask1, yellow)
            combined_mask2 = cv2.bitwise_and(mask2, yellow)

        if save : 
            debug[str(index) + '-color1'] = combined_mask1
            debug[str(index) + '-color2'] = combined_mask2

        sum1 = np.sum(combined_mask1) / 255
        sum2 = np.sum(combined_mask2) / 255

        dist1 = OrientationComputation.__estimate_min_distance_along_rectangle(box1,distance)
        dist2 = OrientationComputation.__estimate_min_distance_along_rectangle(box2, distance)

        if save : 
            display1 = cv2.bitwise_and(image, image, mask=mask1)
            cv2.putText(display1,str(sum1) + "," + str(dist1),(int(x),int(y)),cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)
            debug[str(index) + '-mask1'] = display1
            display2 = cv2.bitwise_and(image, image, mask=mask2)
            cv2.putText(display2,str(sum2) + "," + str(dist2),(int(x),int(y)),cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)
            debug[str(index) + '-mask2'] = display2
            dst = cv2.normalize(distance, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            display3 = cv2.bitwise_and(dst, dst, mask=mask1)
            cv2.putText(display3,str(sum1) + "," + str(dist1),(int(x),int(y)),cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)
            debug[str(index) + '-dist1'] = display3
            display4 = cv2.bitwise_and(dst, dst, mask=mask2)
            cv2.putText(display4,str(sum2) + "," + str(dist2),(int(x),int(y)),cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)
            debug[str(index) + '-dist2'] = display4

        angle = degrees(candidate1)
        rect = rotated1
        if 0.9 * sum2 > sum1 and sum1 > 50 : 
            angle = degrees(candidate2)
            rect = rotated2
        elif 0.9 * sum1 > sum2 and sum2 > 50 :
            angle = degrees(candidate1)
            rect = rotated1
        elif dist2 < dist1 :
            angle = degrees(candidate2)
            rect = rotated2

        return angle, rect, debug


    def process(self, image, samples, save = False):
        """
        Process new image
        Parameters :
            image (array)   : The image to process
            samples (array) : The samples data to track. To compy with limelight constraints, it
                is an array containing : x position of the first sample,
                y position of the first sample, ... , x position of the second sample, y position of the
                second samples, ... (see input formatting for more details)
            save (bool)     : true if the pipeline shall save intermediar images
        Returns :
            A dict containing all harvested data, and at least :
                - The display image
                - The pipeline python outputs
                - The pipeline overlay
                - The data for metrics evaluation (optional)
        """

        result = {}
        display = image.copy()

        formatted = self.__format_inputs(samples)

        # Compute image edges
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 80, 200)
        if save :
            result['edges'] = edges
        
        # Compute contours distance map
        edges_inv = cv2.bitwise_not(edges)
        if save :
            result['edges-inv'] = edges_inv
        dist = cv2.distanceTransform(edges_inv, cv2.DIST_L2, 0)
        if save :
            result['distance'] = cv2.normalize(dist, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            
        # Compute hsv image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Compute color masks
        blue = cv2.inRange(hsv, 
                np.array([self.__hue_blue_min, 30, 30]),
                np.array([self.__hue_blue_max, 255, 255]))
        yellow = cv2.inRange(hsv, 
                np.array([self.__hue_yellow_min, 30, 30]),
                np.array([self.__hue_yellow_max, 255, 255]))
        mask1 = cv2.inRange(hsv, 
                np.array([self.__hue_red_min1, 30, 30]),
                np.array([self.__hue_red_max1, 255, 255]))
        mask2 = cv2.inRange(hsv, 
                np.array([self.__hue_red_min2, 30, 30]),
                np.array([self.__hue_red_max2, 255, 255]))
        red = cv2.bitwise_or(mask1, mask2)

        if save :
            result['red'] = red
            result['yellow'] = yellow
            result['blue'] = blue
        
        for i_sample, sample in enumerate(formatted) :

            # For each sample, find the area it belongs to and derive color and orientation
            angle, rect, local = self.__process_area(image, yellow, blue, red, dist, sample['x'], sample['y'], sample['sx'], sample['sy'], sample['color'], sample['index'], save)
            result.update(local)
            sample['angle'] = float(angle)
            sample['rect'] = rect
        
        # Prepare outputs
        outputs = [self.__index]
        for sample in formatted :
            OrientationComputation.__draw_sample(display, sample['color'], sample['angle'], sample['rect'], sample['xmin'], sample['xmax'], sample['ymin'], sample['ymax'])
            outputs.append(sample['index'])
            outputs.append(float(sample['x']))
            outputs.append(float(sample['y']))
            outputs.append(int(sample['color']))
            outputs.append(float(sample['angle']))
            outputs.append(float(sample['area']))
            outputs.append(float(sample['xmin']))
            outputs.append(float(sample['xmax']))
            outputs.append(float(sample['ymin']))
            outputs.append(float(sample['ymax']))
            

        result['overlay'] = np.array([])
        result['display'] = display
        result['outputs'] = outputs

        print(outputs)

        return result
        

PIPELINE = OrientationComputation()
def runPipeline(image, llrobot):
   
    results = PIPELINE.process(image, llrobot)
    return results['overlay'] , results['display'], results['outputs']
