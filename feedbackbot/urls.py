"""feedbackbot URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.without,name="home"),
    path('check/',views.output,name="out"),
    path('botstart/',views.tran_to_botstart,name="botst"),
    path('yes/',views.yes,name="yes"),
    
    path('chec/',views.again,name="again"),
    path('again/',views.tran_to_again,name="aga"),
    path('user/',views.tran_to_username,name="user"),
    path('welcome/',views.logincheck,name="welcome"),
    path('result/',views.spe,name="spe"),
    
    
     
    
]

urlpatterns += staticfiles_urlpatterns()
