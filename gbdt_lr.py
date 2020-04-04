        #gbdt???????
        clf = xgb.XGBRegressor(
            n_estimators=30,#????
            learning_rate =0.1,
            max_depth=3,
            min_child_weight=1,
            gamma=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            reg_lambda=1,
            seed=27)
        model_sklearn=clf.fit(X, y)
        y_sklearn= clf.predict(test_X)
        print('max:',np.expm1(y_sklearn).max())
        train_new_feature= clf.apply(X)#????????????????
        test_new_feature= clf.apply(test_X)
        X = train_new_feature.copy()
        test_X = test_new_feature.copy()

        from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, ElasticNetCV, RidgeCV, LassoLarsCV, BayesianRidge
        from sklearn.model_selection import cross_val_score
        def rmse_cv(model):
            rmse= np.sqrt(-cross_val_score(model, X, np.expm1(y), scoring="neg_mean_squared_error", cv = 5))
            return(rmse)

        from sklearn.neighbors import KNeighborsRegressor
        knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()

        knn.fit(X, np.expm1(y))

        print('pay_price of i',i, 'rmse:', rmse_cv(knn).mean())