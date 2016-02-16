import cv2
import numpy as np
import time as t
import math
import matplotlib.pyplot as plt
from scipy import signal


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('output_g1.avi')
timeg=[]
distance=[]
velocity=[]
g=[]
filt=[]

time=[]
flag_dist=0
prev_distance=0
ldist=0



while(1):
     dist=0
     xlast=0
     ylast=0
     xp=0
     yp=0
     
     
    # Take each frame
     ret, frame1 = cap.read()

     if ret == True:
         # Convert BGR to HSV
        frame=cv2.GaussianBlur(frame1,(5,5),0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       
    # define range of blue color in HSV
    #lower_blue = np.array([110,50,50])
    #upper_blue = np.array([130,255,255])

        lower_blue = np.array([1,141,56])
        upper_blue = np.array([10,190,143])
    
    # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask=cv2.erode(mask,None,iterations=2)
        mask=cv2.dilate(mask,None,iterations=2)
        img_th = cv2.adaptiveThreshold(mask.copy(),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #contours = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        _,contours, hierarchy = cv2.findContours(img_th.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        #print contours
        
        #print M

        if len(hierarchy)>0:

            cnt = contours[0]
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)
            #print area
            #print perimeter

            #c=max(contours,key=cv2.contourArea)
            ((x,y),radius)=cv2.minEnclosingCircle(cnt)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            xp=center[0]
            yp=center[1]
            
            if (xp-xlast>=23 and yp-ylast>=23):
                 
                 
                 dist=math.sqrt(((xp-xlast)**2)+((yp-ylast)**2))
                 #print dist
                 if (flag_dist==0):
                      prev_distance=dist
                      flag_dist+=1
            
                 if ((dist-ldist)>1):
                      
                      #print dist,
                      #print ldist
                      distance.append((dist-prev_distance)*(0.6/121))
                      time.append(cap.get(0))
                      ldist=dist

                 
            xlast=xp
            ylast=yp
            
            
            
            #print prev_distance
            #print distance
           
            #print cap.get(0)
            
            
            
            
            

        if radius > 0:
            cv2.circle(frame,(int(x),int(y)),int (radius),(0,255,255),2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
            
    

    
    
    
    

    # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
    

        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            
            break
        
     else:
        break
     #t.sleep(0.25)
         

    

cv2.destroyAllWindows()
cap.release()

#print distance
filt=signal.medfilt(distance)


maxdist_index=np.argmax(filt)
mindist_index=np.argmin(filt)
max_time = time[maxdist_index]
min_time = time[mindist_index]
print "Distance: "+str(max(filt))+" m"
print "Stop Time: ",
print str(max_time)+ " ms",
print" Start Time: ",
print str(min_time)+" ms"
print " "

acceleration_g=(2*max(filt)*1000000)/((max_time-min_time)**2)
print "Acceleration due to gravity = "+str(acceleration_g)+" m/s2"
velocity=np.gradient(filt)
g=np.gradient(velocity*1000)

timeg=np.gradient(time)

f1=plt.figure(1)
f1.suptitle('Distance v/s time')
plt.plot(time,filt)
plt.ylabel('distance (m)')
plt.xlabel('time(ms)')
plt.show()

f2=plt.figure(2)
f2.suptitle('Velocity v/s time')
plt.plot(time,np.divide(velocity*1000,timeg))
#plt.plot(time,velocity*1000)
plt.ylabel('velocity (m/s)')
plt.xlabel('time(ms)')
plt.show()

f3=plt.figure(3)
f3.suptitle('Acceleration v/s time')
plt.plot(time,np.divide(g,timeg))
plt.ylabel('acceleration (m/s2)')
plt.xlabel('time(ms)')
plt.show()
