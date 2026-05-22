# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from llamafactory.v1.utils.plugin import BasePlugin


def test_register_methods_registers_staticmethod_group():
    class TestPlugin(BasePlugin):
        pass

    @TestPlugin("demo").register_methods()
    class DemoPlugin:
        @staticmethod
        def run(value: int) -> int:
            return value + 1

    assert TestPlugin("demo").run(1) == 2
    assert DemoPlugin.run(1) == 2


def test_register_methods_rejects_class_attribute():
    class TestPlugin(BasePlugin):
        pass

    with pytest.raises(TypeError, match=r"BadPlugin\.engine must be a staticmethod"):

        @TestPlugin("bad").register_methods()
        class BadPlugin:
            engine = None


def test_register_methods_does_not_partially_register_invalid_group():
    class TestPlugin(BasePlugin):
        pass

    with pytest.raises(TypeError, match=r"BadPlugin\.engine must be a staticmethod"):

        @TestPlugin("bad").register_methods()
        class BadPlugin:
            @staticmethod
            def run() -> None:
                pass

            engine = None

    with pytest.raises(ValueError, match=r"Method 'run' of plugin 'bad' is not registered"):
        TestPlugin("bad").run()


def test_register_methods_rejects_instance_method():
    class TestPlugin(BasePlugin):
        pass

    with pytest.raises(TypeError, match=r"BadPlugin\.run must be a staticmethod"):

        @TestPlugin("bad").register_methods()
        class BadPlugin:
            def run(self) -> None:
                pass


def test_register_methods_rejects_classmethod():
    class TestPlugin(BasePlugin):
        pass

    with pytest.raises(TypeError, match=r"BadPlugin\.run must be a staticmethod"):

        @TestPlugin("bad").register_methods()
        class BadPlugin:
            @classmethod
            def run(cls) -> None:
                pass
