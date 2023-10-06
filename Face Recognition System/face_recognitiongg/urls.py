from django.urls import path,include
from . import views
app_name = "face_recognitiongg"
urlpatterns = [
    path('', views.index, name='index'),
    path('students/', views.student_list, name='student_list'),
    path('courses/', views.course_list, name='course_list'),
    path('attendance/<int:course_id>/', views.attendance_list, name='attendance_list'),
]