* We have CommonTerm
    * When building, we need to treat numeric terms different from categorical terms.
    * Numeric terms are subject to mathematical operations.
    * Categorical terms are not.
    * Interactions
        * numeric interactions -> no change, like a regular numeric variable.
        * categorical interactions -> no change, like a regular categorical variable.
        * numeric-categorical interaction -> center by level of the categorical variable.
            * Automatically handled by `.mean(0)`
* Data container names
    * `{name}_data`

* ¿Qué información necesito para construir un term?
    * `bambi.Terms.Term`
    * `bambi.Model`
        * Para la familia -> Puedo en cambio usar atributos de la familia.
        * Para las coordenadas de la respuesta --> Aun no se que hacer
        * Para el termino de respuesta (en realidad su dimensión)
        * Para el atributo `.noncentered` --> No se que hacer
* ¿Qué información necesito para construir distributional components?
    * `bambi.prior.Prior`, para el typing y para decidir como resolver algo.


## Other

* Are coords a property of terms, or something created for PyMC?
    * If we want to make it more independent, it should be something created by the PyMC backend.
        * Of course, the term has to offer all the information it can.

## Como implementarlo?

Tengo components, los components tienen terms.
No me preocupan tanto las transformaciones, dimensiones, etc.
Sino el orden de las operaciones.

### Estrategia A

Una pasada por "tipo de construcción"

1. Crear dimensiones y coordenadas
    i. Itero a través de los terminos de un DistributionalComponent.
    Para cada termino, tengo un diccionario de {dims: coords}
    ¿Donde guardo esos diccionarios para uso posterior?
    ```python
    blocks = {
        term_name: {
            "coords": ...,
            "term": ...,
            "data": ...,
            "rv": ...,
        }
    }
    ```


2. Crear data containers
3. Crear random variables
4. Crear expresiones a partir de random variable y data containers
5. Crear response-level parameters (e.g., los parámetros que van en el modelo observacional)

Pro: Cada paso tiene una tarea bien definida, el código es más limpio y consistente.
Contra: ineficiente? un lio?
- Más facil de mantener?

### Estrategia B

Una pasada por "building block"

Es decir, una pasada por cada distributional component.
Dentro de eso, una pasada por cada termino.

La construcción del término puede hacer hasta 3 cosas:

* Crear una random variable (siempre)
* Crear un data container (siempre?)
* Crear coordenadas y agregarlas al modelo
    * Tengo dudas sobre esto, que tan robusto puede ser el proceso, por ejemplo, cuando tengo que reutilizar coordenadas?


---

* Necesito algun proceso que recorra todos los terminos de un modelo.
Ese proceso debe devolver un diccionario de dims -> coords.

* Primero se crean las dimensiones y coordenadas
* Luego las coordenadas
* Luego 
