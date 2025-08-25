REM Exécuter l’étape de configuration/outil
@echo off
REM Exécuter l’étape de configuration/outil
set TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false
REM Exécuter l’étape de configuration/outil
set TF_DISABLE_JIT=1
REM Exécuter l’étape de configuration/outil
echo XLA disabled for this session.