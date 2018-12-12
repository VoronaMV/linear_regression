import os
import pandas as pd
import numpy as np
from linear import Linear
from utils import argument_parser, print_formula, handle_result


if __name__ == '__main__':

    args = argument_parser()

    if not args.predict:
        if not os.path.isfile(args.file):
            exit(f'No such file {args.file}')
        data = pd.read_csv(args.file, sep=',')
        try:
            linear = Linear(data=data, target_name=args.target_var)
        except Exception as e:
            exit(e)
        weights = linear.fit(learning_rate=args.learning_rate,
                             accuracy=args.accuracy,
                             to_file=args.to_file)
    else:
        if not os.path.isfile(args.model_file):
            exit(f'No such file {args.model_file}')
        if args.predict_from_file:
            if not os.path.isfile(args.predict_from_file):
                exit(f'No such file {args.predict_from_file}')
            data = np.loadtxt(args.predict_from_file, delimiter=',')
        else:
            data = []
            while True:
                value = input('Type digit or Q to exit: ')
                if value in 'Qq':
                    if not data:
                        continue
                    break
                else:
                    if not value.isdigit():
                        continue
                    data.append(int(value))
            data = np.array(data)

        try:
            model_data = Linear.from_file(args.model_file)
            x_shape = model_data['x_shape']
            if data.shape != x_shape:
                data = data.reshape(len(data // len(model_data['independent_vars'])), x_shape[1])
            prediction = Linear.predict(data, model_data)
            print_formula(model_data['weights'].flatten(), coef=model_data['coef'])
            params = {
                model_data['independent_vars'][0]: data.flatten(),
                model_data['dependent_var']: prediction.flatten()
            }
            result_df = handle_result(**params)
        except:
            exit('Bad data')

        if args.plot:
            if not os.path.isfile(args.file):
                exit(f'No such file {args.file}')

            train_data = pd.read_csv(args.file, sep=',')
            Linear.plot(train_data, model_data, predictions_df=result_df)
