from sklearn.decomposition import TruncatedSVD

def reduce_dimensions(X, n_components=2):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(X)
