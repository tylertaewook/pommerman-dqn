def featurize(env, states):
    """
    Converts the states(dict) into list of 1D numpy arrays

    Input:
    - env: gym environment
    - states: list[num_agents, dict(15)] for each agent
    Output:
    - feature: list[num_agents, 372]
    """

    feature = []
    for state in states:
        feature.append(env.featurize(state).tolist())
        # changes to 1D numpy array
    return feature
