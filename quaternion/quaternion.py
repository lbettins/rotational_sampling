import math

"""This class is represents a quaternionic ring, which are numbers of the form
   a + bi + cj + dk. The primary axiom for this ring is i^2 = j^2 = k^2 = ijk = -1,
   where -1 represents the additive inverse of the multiplicative identity. We
   also have a, b, c, and d as members of a field (typically the real numbers
   with standard addition and multiplication)."""
class Quaternion:

    """The constructor for quaternions. With three arguments, we create the quaternion that represents
       the three-dimensional vector of these arguments. With four arguments, it creates the literal
       quaternion.
       For example, given the constructor Quaternion(x,y,z), it would create the object representing
       the quaternion xi + yj + zk, whereas the constructor Quaternion(w,x,y,z) would create the
       quaternion w + xi + yj + zk"""
    def __init__(self, *args):
        if len(args) == 3:
            self.real = 0
            self.imag = args[0]
            self.jmag = args[1]
            self.kmag = args[2]
        elif len(args) == 4:
            self.real = args[0]
            self.imag = args[1]
            self.jmag = args[2]
            self.kmag = args[3]
        else:
            assert False, "Quaternion constructor needs to have either 3 or 4 arguments"

    """Returns a quaternion whose absolute value is equal to 1."""
    def unit(self):
        distance = self.norm()

        return self.multiplyConst(1 / distance)
    
    """Rotates the quaternion "self" around the given vector by theta"""
    def rotation(self, theta, xComp, yComp, zComp):
        rotateQuat = self.encodeAngleVector(theta, xComp, yComp, zComp)
        return self.rotationQuat(rotateQuat)
    
    """Returns the vector encoded by self rotated by rotateQuat.
        Note that this method assumes rotateQuat is a unit quaternion"""
    def rotationQuat(self, rotateQuat):
        inverseQuat = Quaternion(rotateQuat.real, -rotateQuat.imag, -rotateQuat.jmag, -rotateQuat.kmag)

        return rotateQuat.multiply(self).multiply(inverseQuat)
    
    """Adds rQuat to self"""
    def add(self, rQuat):
        realPart = self.real + rQuat.real
        imagPart = self.imag + rQuat.imag
        jmagPart = self.jmag + rQuat.jmag
        kmagPart = self.kmag + rQuat.kmag
        return Quaternion(realPart, imagPart, jmagPart, kmagPart)
    
    """Multiplies "self" by lQuat.
       Returns a quaternion that is equivalent to the left product (lQuat * self)."""
    def lMultiply(self, lQuat):
        realPart = (self.real * lQuat.real) - (self.imag * lQuat.imag) - (self.jmag * lQuat.jmag) - (self.kmag * lQuat.kmag)
        imagPart = (self.real * lQuat.imag) + (self.imag * lQuat.real) - (self.jmag * lQuat.kmag) + (self.kmag * lQuat.jmag)
        jmagPart = (self.real * lQuat.jmag) + (self.imag * lQuat.kmag) + (self.jmag * lQuat.real) - (self.kmag * lQuat.imag)
        kmagPart = (self.real * lQuat.kmag) - (self.imag * lQuat.jmag) + (self.jmag * lQuat.imag) + (self.kmag * lQuat.real)
        return Quaternion(realPart, imagPart, jmagPart, kmagPart)

    """Multiplies "self" by rQuat.
       Returns a quaternion that is equivalent to the right product."""
    def rMultiply(self, rQuat):
        realPart = (self.real * rQuat.real) - (self.imag * rQuat.imag) - (self.jmag * rQuat.jmag) - (self.kmag * rQuat.kmag)
        imagPart = (self.real * rQuat.imag) + (self.imag * rQuat.real) + (self.jmag * rQuat.kmag) - (self.kmag * rQuat.jmag)
        jmagPart = (self.real * rQuat.jmag) - (self.imag * rQuat.kmag) + (self.jmag * rQuat.real) + (self.kmag * rQuat.imag)
        kmagPart = (self.real * rQuat.kmag) + (self.imag * rQuat.jmag) - (self.jmag * rQuat.imag) + (self.kmag * rQuat.real)
        return Quaternion(realPart, imagPart, jmagPart, kmagPart)
    
    """Equivalent to rMultiply."""
    def multiply(self, rQuat):
        return self.rMultiply(rQuat)
    
    """Returns the Cartesian representation of the quaternion xi + yj + zk"""
    def toCartesian(self):
        return [self.imag, self.jmag, self.kmag]
    
    """Multiplies the quaternion self by some real number."""
    def multiplyConst(self, constant):
        realPart = self.real * constant
        imagPart = self.imag * constant
        jmagPart = self.jmag * constant
        kmagPart = self.kmag * constant
        return Quaternion(realPart, imagPart, jmagPart, kmagPart)
    
    """Returns the Euclidean norm (absolute value) of self"""
    def norm(self):
        squares = (self.real * self.real) + (self.imag * self.imag) + (self.jmag * self.jmag) + (self.kmag * self.kmag)
        return math.sqrt(squares)
    
    """Encodes an angle-vector combination into a unit quaternion"""
    def encodeAngleVector(self, theta, xComp, yComp, zComp):
        unitQuat = Quaternion(xComp, yComp, zComp).unit()

        realPart = math.cos(theta / 2)

        # Slight optimization.
        calc = math.sin(theta / 2)

        imagPart = unitQuat.imag * calc
        jmagPart = unitQuat.jmag * calc
        kmagPart = unitQuat.kmag * calc

        rotateQuat = Quaternion(realPart, imagPart, jmagPart, kmagPart)
        return rotateQuat
    
    """Decodes self into an angle-vector combination. The vector will be a unit vector.
       Assumes that theta is not equivalent to 0."""
    def decodeQuaternion(self):
        realPart = self.real
        theta = math.acos(realPart) * 2
        calc = math.sin(theta / 2)
        xComp = self.imag / calc
        yComp = self.jmag / calc
        zComp = self.kmag / calc

        return [theta, xComp, yComp, zComp]
