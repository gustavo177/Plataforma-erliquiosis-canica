from django.contrib import admin
from .models import Pruebas, Canino

@admin.register(Pruebas)
class PruebaAdmin(admin.ModelAdmin):
    list_display = ('canino_id', 'id','tipo_prueba', 'get_diagnostico_display',  'archivo', 'dimensiones')

    def get_diagnostico_display(self, obj):
        return obj.get_diagnostico_display()
    get_diagnostico_display.short_description = 'Diagnóstico'

    def canino_id(self, obj):
        return obj.canino.id if obj.canino else 'No asociado'  # Devuelve el ID del canino o un mensaje si es null
    canino_id.short_description = 'ID Canino'  # Nombre que aparecerá en la columna

class CaninoAdmin(admin.ModelAdmin):
    list_display = ('id', 'imagen','id_canino', 'nombre_canino', 'nombre_dueño', 'dictamen', 'cedula', 'telefono', 'años_canino')
    search_fields = ('nombre_canino', 'nombre_dueño', 'cedula')
admin.site.register(Canino, CaninoAdmin)