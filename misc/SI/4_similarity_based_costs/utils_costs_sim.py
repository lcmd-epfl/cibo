from scipy.spatial import distance

def find_min_max_distance_and_ratio_scipy(x, vectors):
    """
    #FUNCTION concerns subfolder 3_similarity_based_costs
    (helper function for get_batch_price function)
    Calculate the minimum and maximum distance between a vector x and a set of vectors vectors.
    Parameters:
        x (numpy.ndarray): The vector x.
        vectors (numpy.ndarray): The set of vectors.
    Returns:
        tuple: The ratio between the minimum and maximum distance, the minimum distance, and the maximum distance.

    Equation for computation of the ratio:
    \[
    p(x, \text{vectors}) = \frac{\min_{i} d(x, \text{vectors}[i])}{\max \left( \max_{i,k} d(\text{vectors}[i], \text{vectors}[k]), \max_{i} d(x, \text{vectors}[i]) \right)}
    \]

    \[
    d(a, b) = \sqrt{\sum_{j=1}^{n} (a[j] - b[j])^2}
    \]
    """
    # Calculate the minimum distance between x and vectors using cdist
    dist_1 = distance.cdist([x], vectors, "euclidean")
    min_distance = np.min(dist_1)
    # Calculate the maximum distance among all vectors and x using cdist
    pairwise_distances = distance.cdist(vectors, vectors, "euclidean")
    max_distance_vectors = np.max(pairwise_distances)
    max_distance_x = np.max(dist_1)
    max_distance = max(max_distance_vectors, max_distance_x)
    # Calculate the ratio p = min_distance / max_distance
    p = min_distance / max_distance
    return p



def get_batch_price(X_train, costy_mols):
    """
    #FUNCTION concerns subfolder 3_similarity_based_costs
    Computes the total price of a batch of molecules.
    to update the price dynamically as the batch is being constructed
    for BO with synthesis at each iteration

    Parameters:
        X_train (numpy.ndarray): The training data.
        costy_mols (numpy.ndarray): The batch of molecules.
    Returns:
        float: The total price of the batch.

    e.g. if a molecule was included in the training set its price will be 0
    if a similar molecule was not included in the training set its price will be 1
    for cases in between the price will be between 0 and 1
    this is done for all costly molecules in the batch and the total price is returned
    """

    X_train_cp = cp.deepcopy(X_train)
    batch_price = 0

    for mol in costy_mols:
        costs = find_min_max_distance_and_ratio_scipy(mol, X_train_cp)
        batch_price += costs  # Update the batch price
        X_train_cp = np.vstack((X_train_cp, mol))

    return batch_price
