"""  Author: Kiran Pattanashetty
   Vision based 'g' value caluclation v1
   GitHub: https://github.com/kirancps/G_opencv.git
   
This program is developed to calculate acceleration due to gravity
of free falling object. The program uses opencv, numpy, scipy and matplotlib
libraries. The accuracy of the results have to be imporved"""



""" importing libraries"""
import cv2
import numpy as np
import time as t
import math
import matplotlib.pyplot as plt
from scipy import signal



#opening video file
cap = cv2.VideoCapture('output_g1.avi')
#cap = cv2.VideoCapture('output_g2.avi')

#variable declarations
timeg=[] #time gradient (for velocity)
distance=[] #stores distance travelled by object"
velocity=[] #stores velocity travelled by object

g=[] #acceleration

filt=[]#stores filtered distance values(median)

time=[]#captures time when object is detected
flag_dist=0
prev_distance=0 #first time distance when it is detected
ldist=0

true_distance=[] #Estimated distance using s=0.5gt^2
true_velocity=[] #estimated velocity



while(1):
     dist=0
     xlast=0
     ylast=0
     
     xp=0
     yp=0
     
     
    # Take each frame
     ret, frame1 = cap.read()

     if ret == True:

         #guassian filter
        frame=cv2.GaussianBlur(frame1,(5,5),0)
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       
   
    
   
         # define range of red color in HSV 
        lower_blue = np.array([1,141,56])
        upper_blue = np.array([10,190,143])
    
    # Threshold the HSV image to get only red color (color of ball is red
    
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        #dialate and erode operation
        mask=cv2.erode(mask,None,iterations=2)
        mask=cv2.dilate(mask,None,iterations=2)
        
        img_th = cv2.adaptiveThreshold(mask.copy(),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
   
        _,contours, hierarchy = cv2.findContours(img_th.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        
        
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

time[:]=[x-min_time for x in time]

for i in time:
     true_distance.append((0.5*9.7779*i*i)*(10**-6))


#print true_distance
true_velocity=np.gradient(true_distance)

print "Distance: "+str(max(filt))+" m"
print "Stop Time: ",
print str(max_time-min_time)+ " ms",
print" Start Time: ",
print str(min_time-min_time)+" ms"
print " "

acceleration_g=(2*max(filt)*1000000)/((max_time-min_time)**2)
print "Acceleration due to gravity = "+str(acceleration_g)+" m/s2"


velocity=np.gradient(filt)

max_velocity_index=np.argmax(velocity)

velocity=velocity[:max_velocity_index]
min_vel=min(velocity)
velocity[:]=[x-min_vel for x in velocity]

true_velocity=true_velocity[:max_velocity_index]
min_true_vel=min(true_velocity)

true_velocity[:]=[y-min_true_vel for y in true_velocity]




g=np.gradient(velocity*1000)


timeg=np.gradient(time)

timeg_vel=timeg[:max_velocity_index]

timeVel=time[:max_velocity_index]



f1=plt.figure(1)
f1.suptitle('Distance v/s Time')
plt.plot(time,filt,'r',label="Actual Path")
plt.plot(time,true_distance,'b',label="Desired Path",linestyle='--')
plt.ylabel('Distance (m)')
plt.xlabel('Time(ms)')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
plt.show()

f2=plt.figure(2)
f2.suptitle('Velocity v/s Time')
plt.plot(timeVel,np.divide(velocity*1000,timeg_vel),'r',label="Actual Velocity")
plt.plot(timeVel,np.divide(true_velocity*1000,timeg_vel),'b',label="Desired Velocity",linestyle='--')

#plt.plot(time,velocity*1000)
plt.ylabel('Velocity (m/s)')
plt.xlabel('Time(ms)')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
plt.show()


