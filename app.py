from flask import Flask, render_template, request
import sympy as sp
import numpy as np
import json

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():

    result = None
    steps = []
    solution_steps = []
    graphJSON = None
    latex_function = None
    latex_derivative = None

    if request.method == 'POST':

        expr = request.form['function']
        x0 = float(request.form['x0'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])

        x = sp.symbols('x')

        try:

            # Symbolic expression
            f_expr = sp.sympify(expr)

            # Derivative
            df_expr = sp.diff(f_expr, x)

            # Latex conversion
            latex_function = sp.latex(f_expr)
            latex_derivative = sp.latex(df_expr)

            # Numeric functions
            f = sp.lambdify(x, f_expr, 'numpy')
            df = sp.lambdify(x, df_expr, 'numpy')

            xn = x0

            for i in range(max_iter):

                fx = f(xn)
                dfx = df(xn)

                if dfx == 0:
                    result = "Derivative became zero."
                    break

                x_new = xn - fx / dfx

                error = abs(x_new - xn)

                # Symbolic substitutions
                f_sub = sp.latex(
                    f_expr.subs(x, sp.Symbol(f"({xn:.6f})"))
                )

                df_sub = sp.latex(
                    df_expr.subs(x, sp.Symbol(f"({xn:.6f})"))
                )

                latex_step = f"""
                <h3>Iteration {i}</h3>

                \\[
                x_{{{i+1}}}
                =
                x_{{{i}}}
                -
                \\frac{{f(x_{{{i}}})}}{{f'(x_{{{i}}})}}
                \\]

                <br>

                <b>Substitute into the function:</b>

                \\[
                f({xn:.6f})
                =
                {f_sub}
                \\]

                \\[
                f({xn:.6f})
                =
                {fx:.6f}
                \\]

                <br>

                <b>Substitute into the derivative:</b>

                \\[
                f'({xn:.6f})
                =
                {df_sub}
                \\]

                \\[
                f'({xn:.6f})
                =
                {dfx:.6f}
                \\]

                <br>

                <b>Apply Newton-Raphson Formula:</b>

                \\[
                x_{{{i+1}}}
                =
                {xn:.6f}
                -
                \\frac{{{fx:.6f}}}{{{dfx:.6f}}}
                \\]

                \\[
                x_{{{i+1}}}
                =
                {x_new:.6f}
                \\]

                <hr>
                """

                solution_steps.append(latex_step)

                steps.append({
                    'iteration': i,
                    'x': round(xn, 6),
                    'fx': round(float(fx), 6),
                    'dfx': round(float(dfx), 6),
                    'x_new': round(float(x_new), 6),
                    'error': round(float(error), 6)
                })

                if error < tol:
                    result = round(float(x_new), 10)
                    break

                xn = x_new

            # Graph
            x_vals = np.linspace(x0 - 10, x0 + 10, 400)
            y_vals = f(x_vals)

            graph_data = [
                {
                    'x': x_vals.tolist(),
                    'y': y_vals.tolist(),
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'f(x)'
                },
                {
                    'x': [result],
                    'y': [0],
                    'type': 'scatter',
                    'mode': 'markers',
                    'name': 'Root'
                }
            ]

            graphJSON = json.dumps(graph_data)

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template(
        'newton.html',
        result=result,
        steps=steps,
        graphJSON=graphJSON,
        latex_function=latex_function,
        latex_derivative=latex_derivative,
        solution_steps=solution_steps
    )

if __name__ == '__main__':
    app.run()