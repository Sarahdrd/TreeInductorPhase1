class Node:
    def __init__(self,
                 feature=None,
                 threshold=None,
                 left=None,
                 right=None,
                 label=None,
                 gain=None,
                 samples=None,
                 class_distribution=None):

        """
        Representa un nodo del árbol.
        Si label != None → hoja.
        Si label == None → nodo interno.
        """

        self.feature = feature              # índice de feature usada para el split
        self.threshold = threshold          # valor del threshold
        self.left = left                    # hijo izquierdo
        self.right = right                  # hijo derecho
        self.label = label                  # clase si es hoja
        self.gain = gain                    # valor del criterio de split
        self.samples = samples              # número de muestras en nodo
        self.class_distribution = class_distribution  # conteo por clases

        ### Agregados para compatibilidad ###
        self.is_leaf = (label is not None)  # bool, NO método
        self.leaf_value = label             # pruning usa leaf_value
        self.feat_idx = feature             # pruning usa feat_idx
        self.majority_class = None          # REP & PEP la usan

        self.left_weight = 0.5
        self.right_weight = 0.5

        # Si tenemos distribución de clases, definimos major_class
        if class_distribution is not None and len(class_distribution) > 0:
            self.majority_class = class_distribution.argmax()
        else:
            self.majority_class = label     # fallback seguro

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(label={self.label})"
        return f"Node(feature={self.feature}, threshold={self.threshold})"
