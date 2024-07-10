#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <vector>

#include <balsa.h>

using namespace balsa;

////////////////////////////////////////////////////////////////////////////////
// Python docstrings.
////////////////////////////////////////////////////////////////////////////////

/* clang-format off */
PyDoc_STRVAR( balsa_py_doc, R"###(
Balsa
=====

A Python interface for the balsa C++ library for random forest classification.

The interface consists of a package-level function `train` to train random
forests, and a class `RandomForestClassifier` that can be used to classify one
or more data sets using a pre-trained random forest.

For documentation, see the docstings as well as the documentation of the balsa
C++ library.)###" );

PyDoc_STRVAR( balsa_train_py_doc, \
R"###(Trains a random forest given an array of data points and an array of
ground-truth labels.

Parameters
----------
data : array-like
    Array of data points of shape [No. of data points, No. of features]. A copy
    will be made unless the input is a 2-dimensional row-major (C-order),
    contiguous, data type aligned array of a supported data type. Supported data
    types are 32-bit floats (`np.float32`) and 64-bit floats (`np.float64`).
    Arrays of an unsupported data type will be converted to the smallest
    supported data type to which conversion is possible without causing
    truncation or rounding.
labels : array-like
    Array of labels of shape [No. of data points]. A copy will be made unless
    the input is a 1-dimensional row-major (C-order), contiguous, data type
    aligned array of the 8-bit unsigned integers.
model_filename : path-like
    Name (or path) of the file in which the trained forest will be stored.
features_to_consider : int, optional, default = 0
    Number of features to consider when splitting a node. When a node is to be
    split, the specified number of features will be randomly selected from the
    set of all features, and the optimal location for the split will be
    determined using the selected features. If no valid split can be found,
    then features that were initially skipped will be considered as well.
    Effectively, this parameter sets the minimum number of features that will
    be considered. More features will be considered if necessary to find a
    valid split. If set to zero, the square root of the number of features will
    be used(rounded down).
max_depth : int, optional, default = 4294967295
    Maximum distance from any node to the root of the tree.
min_purity : float, optional, default = 1.0
    Minimum Gini-purity to reach. When the purity of a node reaches this
    minimum, the node will not be split further. A minimum purity of 1.0 (the
    default) means nodes will be split until all remaining data points in a node
    have the same label. The minimum possible Gini-purity for any node in a
    classification problem with M labels is 1/M. Setting the minimum purity to
    this number or lower means no nodes will be split at all.
tree_count : int, optional, default = 150
    Number of decision trees that will be trained.
concurrent_trainers : int, optional, default = 1
    Number of decision trees that will be trained concurrently.

Returns
-------
`None`; The trained random forest will be stored as a binary file (see the
`model_filename` parameter).
)###" );

PyDoc_STRVAR( RandomForestClassifier_py_doc, \
R"###(Creates a random forest classifier from an existing model file.

Parameters
----------
model_filename : path-like
    Name (or path) of the file that contains the pre-trained random forest.
max_threads : int, optional, default = 0
    Number of worker threads used for classification. Set to zero (the default)
    for single threaded classification.
max_preload : int, optional, default = 1
    Number of decision trees to pre-load. Balsa can pre-load decision trees in
    batches during classification, instead of reading the entire random forest
    in memory all at once. This significantly reduces the amount of memory
    needed. However, the decision trees will need to be reread for each call to
    `classify`, which takes time. If enough memory is available, it is
    therefore best to read the entire random forest into memory
    (`max_preload` set to zero). Otherwise, set `max_preload` to a multiple of
    the number of threads used for classification.

Returns
-------
A random forest classifier object.
)###" );

PyDoc_STRVAR( RandomForestClassifier_get_class_count_py_doc, \
R"###(Returns the number of classes distinguished by the random forest
classifier.

Parameters
----------
None.

Returns
-------
The number of classes distinguished by the classifier.
)###" );

PyDoc_STRVAR( RandomForestClassifier_get_feature_count_py_doc, \
R"###(Returns the number of features expected by the random forest classifier.

Parameters
----------
None.

Returns
-------
The number of features expected by the random forest classifier.
)###" );

PyDoc_STRVAR( RandomForestClassifier_set_class_weights_py_doc, \
R"###(Set the relative weight of each class.

Parameters
----------
class_weights : array-like
    Array of class weights. Class weights are multiplication factors that will
    be applied to each class vote total before determining the maximum score and
    final label. All class weights must be non-negative (>=0) and the number of
    weights provided must equal the number of classes distinguished by the
    classifier.

Returns
-------
Nothing.
)###" );

PyDoc_STRVAR( RandomForestClassifier_classify_py_doc, \
R"###(Classifies an array of data points using a pre-trained random forest
classifier.

Parameters
----------
data : array-like
    Array of data points of shape [No. of data points, No. of features]. A copy
    will be made unless the input is a 2-dimensional row-major (C-order),
    contiguous, data type aligned array of a supported data type. Supported data
    types are 32-bit floats (`np.float32`) and 64-bit floats (`np.float64`).
    Arrays of an unsupported data type will be converted to the smallest
    supported data type to which conversion is possible without causing
    truncation or rounding. The number of features in `data` must be identical
    to the number of features expected by the random forest classifier.

Returns
-------
An array of shape [No. of data points] with the predicted labels.
)###" );

/* clang-format on */

////////////////////////////////////////////////////////////////////////////////
// Helper functions.
////////////////////////////////////////////////////////////////////////////////

// Attempt to convert a Python object to a 2-D, C contiguous, aligned NumPy
// array of floats or doubles. Only conversions that will not cause truncation,
// rounding, or other changes will be performed (if necessary).
static PyArrayObject * data_array_from_object( PyObject * data_py_object )
{
    // Try to convert the provided Python object to a 2-D NumPy array without
    // any further restrictions.
    PyArrayObject * data_py_array = (PyArrayObject *) PyArray_FromAny( data_py_object, NULL, 2, 2, 0, NULL );
    if ( data_py_array == NULL )
    {
        return NULL;
    }

    // Convert the unrestricted 2-D NumPy array to a 2-D, C contiguous, aligned
    // NumPy array of floats, if possible. Otherwise, attempt a conversion to a
    // 2-D, C contiguous, aligned NumPy array of doubles.
    PyArrayObject * result      = NULL;
    PyArray_Descr * descr_float = PyArray_DescrFromType( NPY_FLOAT );
    if ( PyArray_CanCastArrayTo( data_py_array, descr_float, NPY_SAFE_CASTING ) )
    {
        result = (PyArrayObject *) PyArray_FromArray( data_py_array, PyArray_DescrFromType( NPY_FLOAT ), NPY_ARRAY_IN_ARRAY );
    }
    else
    {
        result = (PyArrayObject *) PyArray_FromArray( data_py_array, PyArray_DescrFromType( NPY_DOUBLE ), NPY_ARRAY_IN_ARRAY );
    }

    // Clean up.
    Py_DECREF( descr_float );
    Py_DECREF( data_py_array );
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// RandomForestClassifier Python type.
////////////////////////////////////////////////////////////////////////////////

/* clang-format off */
// RandomForestClassifier Python type (derives from PyObject via PyObject_HEAD).
typedef struct
{
    PyObject_HEAD
    RandomForestClassifier * m_classifier;
} RandomForestClassifier_py_object;

/* clang-format on */

// Forward declarations.
static PyObject * RandomForestClassifier_py_type_new( PyTypeObject * subtype, PyObject * args, PyObject * kwds );
static int        RandomForestClassifier_py_type_init( PyObject * self, PyObject * args, PyObject * kwds );
static void       RandomForestClassifier_py_type_dealloc( PyObject * self );
static PyObject * RandomForestClassifier_get_class_count( RandomForestClassifier_py_object * self, PyObject * args );
static PyObject * RandomForestClassifier_get_feature_count( RandomForestClassifier_py_object * self, PyObject * args );
static PyObject * RandomForestClassifier_set_class_weights( RandomForestClassifier_py_object * self, PyObject * args );
static PyObject * RandomForestClassifier_classify( RandomForestClassifier_py_object * self, PyObject * args );

// Methods of type RandomForestClassifier.
static PyMethodDef RandomForestClassifier_py_type_methods[] = {
    { "get_class_count", (PyCFunction) RandomForestClassifier_get_class_count, METH_NOARGS, RandomForestClassifier_get_class_count_py_doc },
    { "get_feature_count", (PyCFunction) RandomForestClassifier_get_feature_count, METH_NOARGS, RandomForestClassifier_get_feature_count_py_doc },
    { "set_class_weights", (PyCFunction) RandomForestClassifier_set_class_weights, METH_O, RandomForestClassifier_set_class_weights_py_doc },
    { "classify", (PyCFunction) RandomForestClassifier_classify, METH_O, RandomForestClassifier_classify_py_doc },
    { NULL, NULL } };

// Optional functionality for type RandomForestClassifier.
static PyType_Slot RandomForestClassifier_py_type_slots[] = {
    { Py_tp_new, (void *) RandomForestClassifier_py_type_new },
    { Py_tp_init, (void *) RandomForestClassifier_py_type_init },
    { Py_tp_dealloc, (void *) RandomForestClassifier_py_type_dealloc },
    { Py_tp_methods, RandomForestClassifier_py_type_methods },
    { Py_tp_doc, (void *) RandomForestClassifier_py_doc },
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
        py_object->m_classifier = NULL;
    }
    return (PyObject *) py_object;
}

// Intializes a newly created instance of the RandomForestClassifier Python
// type.
static int RandomForestClassifier_py_type_init( PyObject * self, PyObject * args, PyObject * kwargs )
{
    // Placeholders for converted arguments.
    PyObject *   model_filename_py_object = NULL;
    unsigned int max_threads              = 0;
    unsigned int max_preload              = 1;

    // Convert positional and keyword arguments.
    static const char * keywords[] = { "", "max_threads", "max_preload", NULL };
    if ( !PyArg_ParseTupleAndKeywords( args, kwargs, "O&|$II", (char **) keywords, PyUnicode_FSConverter, &model_filename_py_object, &max_threads, &max_preload ) )
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

    // Allocate memory for a RandomForestClassifier C++ object.
    py_object->m_classifier = (RandomForestClassifier *) PyObject_Malloc( sizeof( RandomForestClassifier ) );
    if ( py_object->m_classifier == NULL )
    {
        PyErr_SetNone( PyExc_MemoryError );
        return -1;
    }

    // Construct the RandomForestClassifier C++ object using placement new.
    try
    {
        new ( py_object->m_classifier ) RandomForestClassifier( model_filename, max_threads, max_preload );
    }
    catch ( ... )
    {
        PyObject_Free( py_object->m_classifier );
        py_object->m_classifier = NULL;
        Py_DECREF( model_filename_py_object );
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
    if ( py_object->m_classifier != NULL )
    {
        py_object->m_classifier->~RandomForestClassifier();
        PyObject_Free( py_object->m_classifier );
        py_object->m_classifier = NULL;
    }

    // Destroy the RandomForestClassifier Python instance.
    PyTypeObject * py_type = Py_TYPE( self );
    py_type->tp_free( self );
    Py_DECREF( py_type );
}

// Return the number of classes distinguished by the random forest classifier.
static PyObject * RandomForestClassifier_get_class_count( RandomForestClassifier_py_object * self, PyObject * Py_UNUSED( args ) )
{
    return Py_BuildValue( "I", self->m_classifier->getClassCount() );
}

// Return the number of features expected by the random forest classifier.
static PyObject * RandomForestClassifier_get_feature_count( RandomForestClassifier_py_object * self, PyObject * Py_UNUSED( args ) )
{
    return Py_BuildValue( "I", self->m_classifier->getFeatureCount() );
}

// Set the relative weight of each class.
static PyObject * RandomForestClassifier_set_class_weights( RandomForestClassifier_py_object * self, PyObject * args )
{
    // Convert the class weights array argument to a 1-D, C contiguous, aligned
    // NumPy array of doubles. This may require a copy of the underlying data.
    PyArrayObject * class_weights_py_array = (PyArrayObject *) PyArray_FROMANY( args, NPY_FLOAT, 1, 1, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST );
    if ( class_weights_py_array == NULL )
    {
        return NULL;
    }

    // Extract the number of classes from the shape of the class weights array.
    const npy_intp number_of_classes_py = PyArray_SIZE( class_weights_py_array );
    assert( number_of_classes_py >= 0 );
    if ( number_of_classes_py > UINT_MAX )
    {
        PyErr_SetString( PyExc_ValueError, "Too many class weights provided." );
        Py_DECREF( class_weights_py_array );
        return NULL;
    }
    const unsigned int number_of_classes = (unsigned int) number_of_classes_py;

    // Check the number of classes against the number of classes distinguished by the classifier.
    if ( number_of_classes != self->m_classifier->getClassCount() )
    {
        PyErr_SetString( PyExc_ValueError, "The number of class weights provided should equal the number of classes distinguished by the classifier." );
        Py_DECREF( class_weights_py_array );
        return NULL;
    }

    // Ensure that all class weights are non-negative.
    for ( unsigned int i = 0; i < number_of_classes; ++i )
    {
        if ( *(float *) PyArray_GETPTR1( class_weights_py_array, i ) < 0.0f )
        {
            PyErr_SetString( PyExc_ValueError, "Class weights must be non-negative." );
            Py_DECREF( class_weights_py_array );
            return NULL;
        }
    }

    // Set the class weights.
    try
    {
        std::vector<float> class_weights;
        for ( unsigned int i = 0; i < number_of_classes; ++i )
        {
            class_weights.push_back( *(float *) PyArray_GETPTR1( class_weights_py_array, i ) );
        }
        self->m_classifier->setClassWeights( class_weights );
    }
    catch ( const Exception & exception )
    {
        PyErr_SetString( PyExc_RuntimeError, exception.getMessage().c_str() );
        Py_DECREF( class_weights_py_array );
        return NULL;
    }
    catch ( ... )
    {
        PyErr_SetString( PyExc_RuntimeError, "Internal error." );
        Py_DECREF( class_weights_py_array );
        return NULL;
    }

    // Clean up.
    Py_DECREF( class_weights_py_array );
    Py_RETURN_NONE;
}

// Classify a set of data points.
static PyObject * RandomForestClassifier_classify( RandomForestClassifier_py_object * self, PyObject * args )
{
    // Convert the data array argument to a NumPy array with a supported data
    // type (float or double), expected memory layout, and number of
    // dimensions. This may require a copy of the underlying data.
    PyArrayObject * data_py_array = data_array_from_object( args );
    if ( data_py_array == NULL )
    {
        return NULL;
    }

    // Extract the number of features from the shape of the data array.
    const npy_intp number_of_features_py = PyArray_DIM( data_py_array, 1 );
    assert( number_of_features_py >= 0 );
    if ( number_of_features_py > UINT_MAX )
    {
        PyErr_SetString( PyExc_ValueError, "The input data contains too many features." );
        Py_DECREF( data_py_array );
        return NULL;
    }
    const unsigned int number_of_features = (unsigned int) number_of_features_py;

    // Check the number of features against the number expected by the classifier.
    if ( number_of_features != self->m_classifier->getFeatureCount() )
    {
        PyErr_SetString( PyExc_ValueError, "The number of features in the input data differs from the number of features expected by the classifier." );
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
        const int data_type = PyArray_TYPE( data_py_array );
        if ( data_type == NPY_FLOAT )
        {
            const float * data_begin   = (const float *) PyArray_DATA( data_py_array );
            const float * data_end     = data_begin + PyArray_SIZE( data_py_array );
            uint8_t *     labels_begin = (uint8_t *) PyArray_DATA( labels_py_array );
            self->m_classifier->classify( data_begin, data_end, labels_begin );
        }
        else if ( data_type == NPY_DOUBLE )
        {
            const double * data_begin   = (const double *) PyArray_DATA( data_py_array );
            const double * data_end     = data_begin + PyArray_SIZE( data_py_array );
            uint8_t *      labels_begin = (uint8_t *) PyArray_DATA( labels_py_array );
            self->m_classifier->classify( data_begin, data_end, labels_begin );
        }
        else
        {
            // This statement should never be reached.
            assert( false );
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
    double       min_purity               = 1.0;
    unsigned int tree_count               = 150;
    unsigned int concurrent_trainers      = 1;
    unsigned int features_to_consider     = 0;

    // Convert positional and keyword arguments.
    static const char * keywords[] = { "", "", "", "features_to_consider", "max_depth", "min_purity", "tree_count", "concurrent_trainers", NULL };
    if ( !PyArg_ParseTupleAndKeywords( args, kwargs, "OOO&|$IIdII", (char **) keywords, &data_py_object, &labels_py_object, PyUnicode_FSConverter, &model_filename_py_object, &features_to_consider, &max_depth, &min_purity, &tree_count, &concurrent_trainers ) )
    {
        // Depending on where PyArg_ParseTupleAndKeywords() fails, it may have
        // converted the model filename using PyUnicode_FSConverter. Using
        // Py_XDECREF() covers both cases.
        Py_XDECREF( model_filename_py_object );
        return NULL;
    }

    // Convert the model filename to a C++ string. The std::string constructor
    // makes a copy, so we do not need to worry about the order of the call to
    // std::~string() relative to the PyDECREF() call for
    // model_filename_py_object.
    std::string model_filename = PyBytes_AsString( model_filename_py_object );

    // Convert the data array argument to a NumPy array with a supported data
    // type (float or double), expected memory layout, and number of
    // dimensions. This may require a copy of the underlying data.
    PyArrayObject * data_py_array = data_array_from_object( data_py_object );
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
        const int data_type = PyArray_TYPE( data_py_array );
        if ( data_type == NPY_FLOAT )
        {
            // Instantiate single precision trainer.
            EnsembleFileOutputStream                            output_stream( model_filename );
            RandomForestTrainer<const float *, const uint8_t *> trainer( output_stream, features_to_consider, max_depth, min_purity, tree_count, concurrent_trainers );

            // Train.
            const float *   data_begin   = (const float *) PyArray_DATA( data_py_array );
            const float *   data_end     = data_begin + PyArray_SIZE( data_py_array );
            const uint8_t * labels_begin = (const uint8_t *) PyArray_DATA( labels_py_array );
            trainer.train( data_begin, data_end, number_of_features, labels_begin );
        }
        else if ( data_type == NPY_DOUBLE )
        {
            // Instantiate double precision trainer.
            EnsembleFileOutputStream                             output_stream( model_filename );
            RandomForestTrainer<const double *, const uint8_t *> trainer( output_stream, features_to_consider, max_depth, min_purity, tree_count, concurrent_trainers );

            // Train.
            const double *  data_begin   = (const double *) PyArray_DATA( data_py_array );
            const double *  data_end     = data_begin + PyArray_SIZE( data_py_array );
            const uint8_t * labels_begin = (const uint8_t *) PyArray_DATA( labels_py_array );
            trainer.train( data_begin, data_end, number_of_features, labels_begin );
        }
        else
        {
            // This statement should never be reached.
            assert( false );
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
    "_balsa",
    balsa_py_doc,
    -1,
    balsa_methods };

// Initializes the _balsa Python module.
PyMODINIT_FUNC PyInit__balsa( void )
{
    // Initialize the NumPy C API.
    import_array();
    if ( PyErr_Occurred() )
    {
        return NULL;
    }

    // Create the _balsa extension module.
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
