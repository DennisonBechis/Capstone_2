    import matplotlib.pyplot as plt
    from sklearn import linear_model

    # X is the 10x10 Hilbert matrix

    # #############################################################################
    # Compute paths

    n_alphas = 200
    alphas = np.logspace(-2, 7, n_alphas)

    coefs = []
    for a in alphas:
        ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
        ridge.fit(X_train, y_train)
        coefs.append(ridge.coef_)

    # #############################################################################
    # Display results

    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()
