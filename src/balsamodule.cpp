#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <balsa.h>

////////////////////////////////////////////////////////////////////////////////
// Python docstrings.
////////////////////////////////////////////////////////////////////////////////

PyDoc_STRVAR( balsa_py_doc, "A Python interface for the balsa C++ library for random forest classification." );
PyDoc_STRVAR( balsa_train_py_doc, "Train a random forest given a set of data points and a set of labels." );
PyDoc_STRVAR( RandomForestClassifier_classify_py_doc, "Classify a set of data points." );

////////////////////////////////////////////////////////////////////////////////
// RandomForestClassifier Python type.
////////////////////////////////////////////////////////////////////////////////

// Short-hand typedefs for supported classifier types.
typedef RandomForestClassifier<const float *, uint8_t *>  RandomForestClassifier_float;
typedef RandomForestClassifier<const double *, uint8_t *> RandomForestClassifier_double;

// RandomForestClassifier Python type (derives from PyObject via PyObject_HEAD).
typedef struct
{
    PyObject_HEAD
        RandomForestClassifier_float * m_rfc_float;
    RandomForestClassifier_double *    m_rfc_double;
} RandomForestClassifier_py_object;

// Forward declarations.
static PyObject * RandomForestClassifier_py_type_new( PyTypeObject * subtype, PyObject * args, PyObject * kwds );
static int        RandomForestClassifier_py_type_init( PyObject * self, PyObject * args, PyObject * kwds );
static void       RandomForestClassifier_py_type_dealloc( PyObject * self );
static PyObject * RandomForestClassifier_classify( RandomForestClassifier_py_object * self, PyObject * args );

// Methods of type RandomForestClassifier.
static PyMethodDef RandomForestClassifier_py_type_methods[] = {
    { "classify", (PyCFunction) RandomForestClassifier_classify, METH_VARARGS, RandomForestClassifier_classify_py_doc },
    { NULL, NULL } };

// Optional functionality for type RandomForestClassifier.
static PyType_Slot RandomForestClassifier_py_type_slots[] = {
    { Py_tp_new, (void *) RandomForestClassifier_py_type_new },
    { Py_tp_init, (void *) RandomForestClassifier_py_type_init },
    { Py_tp_dealloc, (void *) RandomForestClassifier_py_type_dealloc },
    { Py_tp_methods, RandomForestClassifier_py_type_methods },
    { 0, 0 } };

// Type specification.
static PyType_Spec RandomForestClassifier_py_type_spec = {
    .name      = "RandomForestClassifier",
    .basicsize = sizeof( RandomForestClassifier_py_object ),
    .itemsize  = 0,
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots     = RandomForestClassifier_py_type_slots };

// Creates a new (uninitialized) instance of the RandomForestClassifier Python
// type.
static PyObject * RandomForestClassifier_py_type_new( PyTypeObject * subtype, PyObject * args, PyObject * kwds )
{
    RandomForestClassifier_py_object * py_object = (RandomForestClassifier_py_object *) subtype->tp_alloc( subtype, 0 );
    if ( py_object != NULL )
    {
        py_object->m_rfc_float  = NULL;
        py_object->m_rfc_double = NULL;
    }
    return (PyObject *) py_object;
}

// Intializes a newly created instance of the RandomForestClassifier Python
// type.
static int RandomForestClassifier_py_type_init( PyObject * self, PyObject * args, PyObject * kwargs )
{
    // Placeholders for converted arguments.
    PyObject *   model_filename_py_object = NULL;
    unsigned int feature_count            = 0;
    unsigned int max_threads              = 0;
    unsigned int max_preload              = 1;
    int          single_precision         = 0;

    // Convert positional and keyword arguments.
    static const char * keywords[] = { "", "", "max_threads", "max_preload", "single_precision", NULL };
    if ( !PyArg_ParseTupleAndKeywords( args, kwargs, "O&I|$IIp", (char **) keywords, PyUnicode_FSConverter, &model_filename_py_object, &feature_count, &max_threads, &max_preload, &single_precision ) )
    {
        // Depending on where PyArg_ParseTupleAndKeywords() fails, it may have
        // converted the model filename using PyUnicode_FSConverter. Using
        // Py_XDECREF() convers both cases.
        Py_XDECREF( model_filename_py_object );
        return -1;
    }

    // Convert the model filename to a C++ string. The std::string constructor
    // makes a copy, so we do not need to worry about the order of the call to
    // std::~string() relative to the PyDECREF() call for
    // model_filename_py_object.
    std::string model_filename = PyBytes_AsString( model_filename_py_object );

    // Get a pointer to self.
    RandomForestClassifier_py_object * py_object = (RandomForestClassifier_py_object *) self;

    // Allocate memory for a RandomForestClassifier C++ object of the requested type.
    if ( single_precision )
    {
        py_object->m_rfc_float = (RandomForestClassifier_float *) PyObject_Malloc( sizeof( RandomForestClassifier_float ) );
        if ( py_object->m_rfc_float == NULL )
        {
            PyErr_SetNone( PyExc_MemoryError );
            return -1;
        }
    }
    else
    {
        py_object->m_rfc_double = (RandomForestClassifier_double *) PyObject_Malloc( sizeof( RandomForestClassifier_double ) );
        if ( py_object->m_rfc_double == NULL )
        {
            PyErr_SetNone( PyExc_MemoryError );
            return -1;
        }
    }

    // Construct the RandomForestClassifier C++ object using placement new.
    try
    {
        if ( single_precision )
        {
            new ( py_object->m_rfc_float ) RandomForestClassifier_float( model_filename, feature_count, max_threads, max_preload );
        }
        else
        {
            new ( py_object->m_rfc_double ) RandomForestClassifier_double( model_filename, feature_count, max_threads, max_preload );
        }
    }
    catch ( ... )
    {
        PyObject_Free( py_object->m_rfc_float );
        PyObject_Free( py_object->m_rfc_double );
        Py_DECREF( model_filename_py_object );
        py_object->m_rfc_double = NULL;
        py_object->m_rfc_float  = NULL;
        PyErr_SetString( PyExc_RuntimeError, "Internal error." );
        return -1;
    }

    // Clean up.
    Py_DECREF( model_filename_py_object );
    return 0;
}

// Destroy an instance of the RandomForestClassifier Python type.
static void RandomForestClassifier_py_type_dealloc( PyObject * self )
{
    // Get a pointer to self.
    RandomForestClassifier_py_object * py_object = (RandomForestClassifier_py_object *) self;

    // Destroy the underlying RandomForestClassifier C++ instance.
    if ( py_object->m_rfc_double != NULL )
    {
        py_object->m_rfc_double->~RandomForestClassifier_double();
        PyObject_Free( py_object->m_rfc_double );
    }
    if ( py_object->m_rfc_float != NULL )
    {
        py_object->m_rfc_float->~RandomForestClassifier_float();
        PyObject_Free( py_object->m_rfc_float );
    }

    // Destroy the RandomForestClassifier Python instance.
    PyTypeObject * py_type = Py_TYPE( self );
    py_type->tp_free( self );
    Py_DECREF( py_type );
}

// Classify a set of data points.
static PyObject * RandomForestClassifier_classify( RandomForestClassifier_py_object * self, PyObject * args )
{
    // Placeholders for converted arguments.
    PyObject * data_py_object = NULL;

    // Convert positional and keyword arguments.
    if ( !PyArg_ParseTuple( args, "O", &data_py_object ) )
    {
        return NULL;
    }

    // Determine the feature value type.
    const bool single_precision = self->m_rfc_float != NULL;

    // Convert the data array argument to a NumPy array with the expected data
    // type, memory layout, and number of dimensions. This may require a copy
    // of the underlying data.
    int             typenum       = single_precision ? NPY_FLOAT : NPY_DOUBLE;
    PyArrayObject * data_py_array = (PyArrayObject *) PyArray_FROMANY( data_py_object, typenum, 2, 2, NPY_ARRAY_IN_ARRAY );
    if ( data_py_array == NULL )
    {
        return NULL;
    }

    // Ensure the number of features in the input data match the number of
    // features the classifier was trained on.
    const std::size_t number_of_features_cpp = single_precision ? self->m_rfc_float->getFeatureCount() : self->m_rfc_double->getFeatureCount();
    const npy_intp    number_of_features_py  = PyArray_DIM( data_py_array, 1 );
    assert( number_of_features_py >= 0 );
    if ( number_of_features_py != number_of_features_cpp )
    {
        PyErr_SetString( PyExc_ValueError, "The number of features in the input data differs from the number of features the classifier was trained on." );
        Py_DECREF( data_py_array );
        return NULL;
    }

    // Create an array to hold the classification labels.
    const npy_intp  labels_py_array_dims = PyArray_DIM( data_py_array, 0 );
    PyArrayObject * labels_py_array      = (PyArrayObject *) PyArray_EMPTY( 1, &labels_py_array_dims, NPY_UINT8, false );
    if ( labels_py_array == NULL )
    {
        Py_DECREF( data_py_array );
        return NULL;
    }

    // Classify the data.
    try
    {
        if ( single_precision )
        {
            const float * data_begin   = (const float *) PyArray_DATA( data_py_array );
            const float * data_end     = data_begin + PyArray_SIZE( data_py_array );
            uint8_t *     labels_begin = (uint8_t *) PyArray_DATA( labels_py_array );
            self->m_rfc_float->classify( data_begin, data_end, labels_begin );
        }
        else
        {
            const double * data_begin   = (const double *) PyArray_DATA( data_py_array );
            const double * data_end     = data_begin + PyArray_SIZE( data_py_array );
            uint8_t *      labels_begin = (uint8_t *) PyArray_DATA( labels_py_array );
            self->m_rfc_double->classify( data_begin, data_end, labels_begin );
        }
    }
    catch ( const Exception & exception )
    {
        PyErr_SetString( PyExc_RuntimeError, exception.getMessage().c_str() );
        Py_DECREF( labels_py_array );
        Py_DECREF( data_py_array );
        return NULL;
    }
    catch ( ... )
    {
        PyErr_SetString( PyExc_RuntimeError, "Internal error." );
        Py_DECREF( labels_py_array );
        Py_DECREF( data_py_array );
        return NULL;
    }

    // Clean up.
    Py_DECREF( data_py_array );
    return (PyObject *) labels_py_array;
}

////////////////////////////////////////////////////////////////////////////////
// Module-level methods.
////////////////////////////////////////////////////////////////////////////////

// Train a random forest given a set of data points and a set of labels.
static PyObject * balsa_train( PyObject * self, PyObject * args, PyObject * kwargs )
{
    // Placeholders for converted arguments.
    PyObject *   data_py_object           = NULL;
    PyObject *   labels_py_object         = NULL;
    PyObject *   model_filename_py_object = NULL;
    unsigned int max_depth                = UINT_MAX;
    unsigned int tree_count               = 1;
    unsigned int concurrent_trainers      = 1;
    unsigned int features_to_scan         = 0;
    int          single_precision         = 0;

    // Convert positional and keyword arguments.
    static const char * keywords[] = { "", "", "", "max_depth", "tree_count", "concurrent_trainers", "features_to_scan", "single_precision", NULL };
    if ( !PyArg_ParseTupleAndKeywords( args, kwargs, "OOO&|$IIIIp", (char **) keywords, &data_py_object, &labels_py_object, PyUnicode_FSConverter, &model_filename_py_object, &max_depth, &tree_count, &concurrent_trainers, &features_to_scan, &single_precision ) )
    {
        // Depending on where PyArg_ParseTupleAndKeywords() fails, it may have
        // converted the model filename using PyUnicode_FSConverter. Using
        // Py_XDECREF() convers both cases.
        Py_XDECREF( model_filename_py_object );
        return NULL;
    }

    // Convert the model filename to a C++ string. The std::string constructor
    // makes a copy, so we do not need to worry about the order of the call to
    // std::~string() relative to the PyDECREF() call for
    // model_filename_py_object.
    std::string model_filename = PyBytes_AsString( model_filename_py_object );

    // Convert the data array argument to a NumPy array with the expected data
    // type, memory layout, and number of dimensions. This may require a copy
    // of the underlying data.
    int             typenum       = single_precision ? NPY_FLOAT : NPY_DOUBLE;
    PyArrayObject * data_py_array = (PyArrayObject *) PyArray_FROMANY( data_py_object, typenum, 2, 2, NPY_ARRAY_IN_ARRAY );
    if ( data_py_array == NULL )
    {
        Py_DECREF( model_filename_py_object );
        return NULL;
    }

    // Convert the labels array argument to a NumPy array with the expected data
    // type, memory layout, and number of dimensions. This may require a copy
    // of the underlying data.
    PyArrayObject * labels_py_array = (PyArrayObject *) PyArray_FROMANY( labels_py_object, NPY_UINT8, 1, 1, NPY_ARRAY_IN_ARRAY );
    if ( labels_py_array == NULL )
    {
        Py_DECREF( data_py_array );
        Py_DECREF( model_filename_py_object );
        return NULL;
    }

    // Ensure that the number of labels matches the number of datapoints.
    if ( PyArray_DIM( labels_py_array, 0 ) != PyArray_DIM( data_py_array, 0 ) )
    {
        PyErr_SetString( PyExc_ValueError, "The number of labels differs from the number of data points." );
        Py_DECREF( labels_py_array );
        Py_DECREF( data_py_array );
        Py_DECREF( model_filename_py_object );
        return NULL;
    }

    // Extract the number of features from the shape of the data array.
    const npy_intp number_of_features_py = PyArray_DIM( data_py_array, 1 );
    assert( number_of_features_py >= 0 );
    if ( number_of_features_py > UINT_MAX )
    {
        PyErr_SetString( PyExc_ValueError, "The input data contains too many features." );
        Py_DECREF( labels_py_array );
        Py_DECREF( data_py_array );
        Py_DECREF( model_filename_py_object );
        return NULL;
    }
    const unsigned int number_of_features = (unsigned int) number_of_features_py;

    // Train the random forest.
    try
    {
        if ( single_precision )
        {
            // Instantiate single precision trainer.
            RandomForestTrainer<const float *, const uint8_t *> trainer( model_filename, max_depth, tree_count, concurrent_trainers, features_to_scan );

            // Train.
            const float *   data_begin   = (const float *) PyArray_DATA( data_py_array );
            const float *   data_end     = data_begin + PyArray_SIZE( data_py_array );
            const uint8_t * labels_begin = (const uint8_t *) PyArray_DATA( labels_py_array );
            trainer.train( data_begin, data_end, labels_begin, number_of_features );
        }
        else
        {
            // Instantiate double precision trainer.
            RandomForestTrainer<const double *, const uint8_t *> trainer( model_filename, max_depth, tree_count, concurrent_trainers, features_to_scan );

            // Train.
            const double *  data_begin   = (const double *) PyArray_DATA( data_py_array );
            const double *  data_end     = data_begin + PyArray_SIZE( data_py_array );
            const uint8_t * labels_begin = (const uint8_t *) PyArray_DATA( labels_py_array );
            trainer.train( data_begin, data_end, labels_begin, number_of_features );
        }
    }
    catch ( const Exception & exception )
    {
        PyErr_SetString( PyExc_RuntimeError, exception.getMessage().c_str() );
        Py_DECREF( labels_py_array );
        Py_DECREF( data_py_array );
        Py_DECREF( model_filename_py_object );
        return NULL;
    }
    catch ( ... )
    {
        PyErr_SetString( PyExc_RuntimeError, "Internal error." );
        Py_DECREF( labels_py_array );
        Py_DECREF( data_py_array );
        Py_DECREF( model_filename_py_object );
        return NULL;
    }

    // Clean up.
    Py_DECREF( labels_py_array );
    Py_DECREF( data_py_array );
    Py_DECREF( model_filename_py_object );
    Py_RETURN_NONE;
}

////////////////////////////////////////////////////////////////////////////////
// Module definition and module initalization function.
////////////////////////////////////////////////////////////////////////////////

// Module-level methods.
static PyMethodDef balsa_methods[] = {
    { "train", (PyCFunction) balsa_train, METH_VARARGS | METH_KEYWORDS, balsa_train_py_doc },
    { NULL, NULL, 0, NULL } };

// Module definition.
static struct PyModuleDef balsa_module = {
    PyModuleDef_HEAD_INIT,
    "balsa",
    balsa_py_doc,
    -1,
    balsa_methods };

// Initializes the balsa Python module.
PyMODINIT_FUNC PyInit_balsa( void )
{
    // Initialize the NumPy C API.
    import_array();
    if ( PyErr_Occurred() )
    {
        return NULL;
    }

    // Create the balsa extension module.
    PyObject * module = PyModule_Create( &balsa_module );
    if ( module == NULL )
    {
        return NULL;
    }

    // Add the RandomForestClassifier type.
    PyObject * type = PyType_FromSpec( &RandomForestClassifier_py_type_spec );
    if ( PyModule_AddObject( module, "RandomForestClassifier", type ) < 0 )
    {
        Py_XDECREF( type );
        Py_DECREF( module );
        return NULL;
    }

    return module;
}
