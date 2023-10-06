from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Student(models.Model):
    #link each student to a sinle user account
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    student_id = models.CharField(max_length=10, unique=True)
    enrolled_year = models.IntegerField()
    status = models.BooleanField(default=False)  # True for present, False for absent

    def __str__(self) -> str:
        return self.name

class Course(models.Model):
    name = models.CharField(max_length=100)
    students = models.ManyToManyField(Student)

    def __str__(self):
        return self.name

class Attendance(models.Model):
    date = models.DateField()
    course = models.ForeignKey(Course,on_delete=models.CASCADE)
    students = models.ManyToManyField(Student)
    def __str__(self):
        return f"{self.date} - {self.course}"
