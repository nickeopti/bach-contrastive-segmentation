import inspect
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Type, TypeVar

T = TypeVar("T")


@dataclass
class ClassArguments:
    class_type: Type[object]
    arguments: Dict[str, Any]


def add_options(argument_parser: ArgumentParser, name: str, options: Sequence[Type[T]]) -> ClassArguments:
    argument_group = argument_parser.add_argument_group(name)

    argument_group.add_argument(f"--{name}", type=str, default=options[0].__name__)
    temp_args, _ = argument_parser.parse_known_args()

    selected_class_name = vars(temp_args)[name]
    selected_class = {c.__name__: c for c in options}[selected_class_name]

    arguments = _get_arguments(selected_class)
    for argument in arguments:
        argument_group.add_argument(
            f"--{argument.name}",
            type=argument.annotation,
            default=argument.default if not argument.default == inspect._empty else None
        )

    temp_args, _ = argument_parser.parse_known_args()
    argument_values = {argument.name: vars(temp_args)[argument.name] for argument in arguments}

    return ClassArguments(selected_class, argument_values)


def _get_arguments(from_object: Type[object], excluded_parameters=('self', 'cls', 'device')):
    signature = inspect.signature(from_object.__init__)
    parameters = {k: p for k, p in signature.parameters.items() if p.kind == p.POSITIONAL_OR_KEYWORD}
    return [parameter for parameter in parameters.values() if parameter.name not in excluded_parameters]


if __name__ == "__main__":
    parser = ArgumentParser()

    import selection
    available_selectors = [
        selection.SingleChannelQuantileSelector,
        selection.MultiChannelQuantileSelector,
    ]

    parser = add_options(parser, "selector", available_selectors)

    args = parser.parse_args()
    print(args)
