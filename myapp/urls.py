from django.urls import path
from . import views
from django.conf.urls import handler404

urlpatterns = [
    path('',views.hello,name='index'),
    path('login/', views.loginview, name='login'),
    path('registro/', views.registroview, name='registro'),
    path('principal/', views.principal, name='principal'),
    path('listado_paciente/', views.listado_paciente, name='listado_paciente'),
    path('eliminar_prueba/<int:prueba_id>/', views.eliminar_prueba, name='eliminar_prueba'),
    path('prueba/', views.prueba, name='prueba'),
    path('Eliminar/', views.Eliminar, name='Eliminar'),
    path('logout/',views.signout, name='logout'),
    path('registro1/', views.registro1, name='registro1'),
    path('formulario/', views.formulario, name='formulario'),
    path('bancodatos/', views.bancodatos, name='bancodatos'),
    path('resultadodatos/', views.resultadodatos, name='resultadodatos'),
    path('resultadospacientes/', views.resultadospacientes, name='resultadospacientes'),
    path('analisis/', views.analisis, name='analisis'),
    
]

handler404 = views.error_404_view



