from django.shortcuts import render, get_object_or_404
from .models import Student, Course, Attendance

def index(request):
    # Your view logic goes here
    return render(request, 'face_recognitiongg/index.html')


def student_list(request):
    students = Student.objects.all()
    return render(request, 'face_recognitiongg/student_list.html', {'students': students})

def course_list(request):
    courses = Course.objects.all()
    return render(request, 'face_recognitiongg/course_list.html', {'courses': courses})

def attendance_list(request, course_id):
    course = get_object_or_404(Course, pk=course_id)
    attendance = Attendance.objects.filter(course=course)
    return render(request, 'face_recognitiongg/attendance_list.html', {'attendance': attendance, 'course': course})
