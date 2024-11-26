from django.db import models

class Canino(models.Model):
    id_canino = models.CharField(max_length=100, unique=True)
    nombre_canino = models.CharField(max_length=100)
    nombre_dueño = models.CharField(max_length=100)
    dictamen = models.CharField(max_length=100)
    cedula = models.CharField(max_length=100)
    telefono = models.CharField(max_length=100)
    años_canino = models.IntegerField()
    imagen = models.ImageField(upload_to='images/')

    def __str__(self):
        return self.nombre_canino

class Pruebas(models.Model):
    TIPO_PRUEBA_CHOICES = [
        ('Aliento', 'Aliento'),
        ('opcion2', 'Opción 2'),
        ('opcion3', 'Opción 3'),
    ]
    DIAGNOSTICO_CHOICES = [
        (0, 'Con Erliquiosis'),
        (1, 'Sin Erliquiosis'),
    ]
    canino = models.ForeignKey(Canino, on_delete=models.CASCADE, null=True,)
    tipo_prueba = models.CharField(max_length=100, choices=TIPO_PRUEBA_CHOICES)
    diagnostico = models.IntegerField(choices=DIAGNOSTICO_CHOICES)
    archivo = models.FileField(upload_to='uploads/')
    dimensiones=models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    
   
    def __str__(self):
        return f'Prueba {self.id} - {self.canino.id_canino}'
    
    