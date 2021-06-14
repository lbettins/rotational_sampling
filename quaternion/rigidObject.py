from quaternion import Quaternion
import numpy as np
import math

# Represents an object with a given number of points that can be rotated in any direction.
# Created to be quick in calculations, but slow in printing out results
class RigidObject:
    def __init__(self, length, arr):
        assert length <= arr.size, "Length is less than the array length"
        self.length = length

        # This will be used to keep track of the product of rotations
        self.rotation = Quaternion(1, 0, 0, 0) # The "identity" rotation
        self.origin = Quaternion(0, 0, 0)
        self.translation = Quaternion(0, 0, 0)

        arr = arr[0:length]
        self.arr = []

        for i in range(length):
            q = arr[i]
            self.arr.append(Quaternion(q[0], q[1], q[2]))

    # Turns the points in the object into an array.
    def to_array(self):
        arr = np.zeros((self.length, 3))

        for i in range(self.length):
            quat = self.arr[i].add(self.origin)
            arr[i][0] = quat.imag
            arr[i][1] = quat.jmag
            arr[i][2] = quat.kmag
        
        return arr

    # Gets the point located at the ith position.
    def get(self, i):
        arr = np.zeros(3)
        quat = self.arr[i].add(self.origin)
        arr[0] = quat.imag
        arr[1] = quat.jmag
        arr[2] = quat.kmag

        return arr
    
    # Rotates the object by theta around the vector (xComp, yComp, zComp) around the object's origin
    def rotate(self, theta, xComp, yComp, zComp):
        newRotation = self.rotation.encodeAngleVector(theta, xComp, yComp, zComp)
        self.rotation = newRotation.multiply(self.rotation)
        for i in range(self.length):
            self.arr[i] = self.arr[i].rotationQuat(newRotation)

    # Rotates the object by theta around the vector (xComp, yComp, zComp) around origin
    def rotate_diff_origin(self, theta, xComp, yComp, zComp, origin):
        self.change_origin(origin)
        self.rotate(theta, xComp, yComp, zComp)
    
    # Changes the origin of rotation to be the array "origin".
    def change_origin(self, origin):
        addQuat = Quaternion(self.origin.imag - origin[0], self.origin.jmag - origin[1], self.origin.kmag - origin[2])
        for i in range(self.length):
            self.arr[i] = self.arr[i].add(addQuat)
        
        self.rotation = Quaternion(1, 0, 0, 0)
        self.origin = Quaternion(origin[0], origin[1], origin[2])

    # Translates the object by the vector (xComp, yComp, zComp)
    def translate(self, xComp, yComp, zComp):
        translation = Quaternion(xComp, yComp, zComp)
        self.translation = self.translation.add(translation)
        self.origin = self.origin.add(translation)
    
    # Rotates the object by theta around the x-axis
    def rotate_x(self, theta):
        self.rotate(theta, 1, 0, 0)
    
    # Rotates the object by theta around the x axis of newly defined origin.
    def rotate_x_diff_origin(self, theta, origin):
        self.rotate_diff_origin(theta, 1, 0, 0, origin)
    
    # Rotates the object by theta around the y-axis
    def rotate_y(self, theta):
        self.rotate(theta, 0, 1, 0)
    
    # Rotates the object by theta around the y-axis of newly defined origin.
    def rotate_y_diff_origin(self, theta, origin):
        self.rotate_diff_origin(theta, 0, 1, 0, origin)
    
    # Rotates the object by theta around the z-axis
    def rotate_z(self, theta):
        self.rotate(theta, 0, 0, 1)

    # Rotates the object by theta around the z-axis of newly defined origin.
    def rotate_z_axis_diff_origin(self, theta, origin):
        self.rotate_diff_origin(theta, 0, 0, 1, origin)
    
    # Returns the angles rotated around the x, y, and z axes that would obtain the same rotation the object is currently at.
    # The rotation identity is the object at the beginning of the definition of self.origin.
    def total_rotate_axes(self):
        arr = np.zeros(3)

        calc1 = 2 * (self.rotation.real * self.rotation.jmag - self.rotation.kmag * self.rotation.imag)
        # Correct minor floating point errors that yield errors.
        if calc1 > 1:
            calc1 = 1
        elif calc1 < -1:
            calc1 = -1

        if (abs(calc1 - 1) < 1e-8):
            arr[0] = 2 * math.atan2(self.rotation.imag, self.rotation.real)
            arr[1] = math.pi / 2
            arr[2] = 0
            return arr
        
        if (abs(calc1 + 1) < 1e-8):
            arr[0] = 2 * math.atan2(self.rotation.imag, self.rotation.real)
            arr[1] = -math.pi / 2
            arr[2] = 0
            return arr

        calc00 = 2 * (self.rotation.real * self.rotation.imag + self.rotation.jmag * self.rotation.kmag)
        calc01 = 1 - (2 * (self.rotation.imag * self.rotation.imag + self.rotation.jmag * self.rotation.jmag))
        
        calc20 = 2 * (self.rotation.real * self.rotation.kmag + self.rotation.imag * self.rotation.jmag)
        calc21 = 1 - (2 * (self.rotation.jmag * self.rotation.jmag + self.rotation.kmag * self.rotation.kmag))

        arr[0] = math.atan2(calc00, calc01)
        arr[1] = math.asin(calc1)
        arr[2] = math.atan2(calc20, calc21)
        return arr
    
    # Returns the vector of rotation as well as the angle of rotation.
    def total_rotate_angle_vec(self):
        q = self.rotation
        return q.decodeQuaternion()
